from __future__ import annotations

import asyncio

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger, setup_logging
from furrow.state import StateManager

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        max_cycles: int = 10,
        resume: bool = False,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.max_cycles = max_cycles
        self.all_tasks: list[TaskModel] = []
        self.state_manager = StateManager(self.client.settings.workspace, goal)
        setup_logging(level=self.client.settings.log_level, json_logs=self.client.settings.json_logs)

        if resume:
            self._resume_state()

    def _resume_state(self) -> None:
        """Load previous state if available."""
        state = self.state_manager.load()
        if state:
            self.all_tasks = self.state_manager.restore_tasks(state)
            self.cycles = state.get("cycles", 0)
            self.goal = state.get("goal", self.goal)
            logger.info("state_resumed", cycles=self.cycles, task_count=len(self.all_tasks))
            console.print(f"[green]Resumed from cycle {self.cycles} with {len(self.all_tasks)} tasks[/green]")

    def _save_state(self) -> None:
        """Save current state."""
        self.state_manager.save(self.all_tasks, self.cycles, self.goal)

    async def run(self) -> None:
        logger.info("furrow_started", goal=self.goal, max_cycles=self.max_cycles)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("cycle_started", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                logger.info("furrow_completed", cycles=self.cycles, total_tasks=len(self.all_tasks))
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                self.state_manager.clear()
                break
            if self.cycles >= self.max_cycles:
                logger.warning("max_cycles_reached", max_cycles=self.max_cycles)
                console.print(f"[yellow]Max cycles ({self.max_cycles}) reached. Stopping.[/yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, self.all_tasks)
        logger.info("plan_created", cycle=self.cycles, task_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            logger.warning("no_tasks_planned", cycle=self.cycles)
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Add new tasks to our tracking list
        for task in plan.tasks:
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
                logger.error("task_failed", task_id=task.id, error=str(result))
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                logger.info("task_completed", task_id=task.id)
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console) as _:
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            logger.info("tests_passed", summary=test_result.summary)
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self.all_tasks:
            return False
        completed = sum(1 for t in self.all_tasks if t.status == "completed")
        failed = sum(1 for t in self.all_tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(self.all_tasks)
