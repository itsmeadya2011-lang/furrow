from __future__ import annotations

import asyncio
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, settings
from furrow.core.state import StateManager
from furrow.llm import LLMClient
from furrow.logging import configure_logging, get_logger

configure_logging(settings.log_level, settings.log_file)
logger = get_logger(__name__)

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 0,
        workspace: Path | None = None,
        resume: bool = False,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.max_cycles = max_cycles
        self.workspace = workspace or Path.cwd()
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.completed_task_ids: set[str] = set()
        self.state_manager = StateManager(self.workspace)
        
        if resume:
            self._load_state()

    def _load_state(self) -> None:
        state = self.state_manager.load()
        if state:
            self.goal = state.get("goal", self.goal)
            self.cycles = state.get("cycles", 0)
            self.completed_task_ids = set(state.get("completed_task_ids", []))
            for task_data in state.get("tasks", []):
                self.all_tasks.append(TaskModel(**task_data))
            logger.info("Resumed from previous state", cycles=self.cycles, task_count=len(self.all_tasks))
            console.print(f"[green]Resumed from cycle {self.cycles} with {len(self.all_tasks)} tasks[/green]")

    def _save_state(self) -> None:
        self.state_manager.save(self.goal, self.cycles, self.all_tasks, self.completed_task_ids)
        logger.debug("State saved", cycles=self.cycles, completed=len(self.completed_task_ids))

    async def run(self) -> None:
        logger.info("Starting Furrow orchestrator", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("Starting cycle", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                logger.info("Goal completed", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self.state_manager.clear()
                break
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                logger.warning("Max cycles reached", max_cycles=self.max_cycles)
                console.print(f"[yellow]Reached max cycles ({self.max_cycles}). Stopping.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        logger.info("Plan created", task_count=len(plan.tasks), rationale=plan.rationale)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("No tasks planned")
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracked list
        for task in plan.tasks:
            if task.id not in self.completed_task_ids and not any(t.id == task.id for t in self.all_tasks):
                self.all_tasks.append(task)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                logger.error("Task failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self.completed_task_ids.add(task.id)
                logger.info("Task completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("Tests passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("Tests failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.id in self.completed_task_ids)
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
