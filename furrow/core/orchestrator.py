from __future__ import annotations

import asyncio
import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel
from furrow.llm import LLMClient
from furrow.logging import get_logger

console = Console()
logger = get_logger("orchestrator")

STATE_FILE = Path(".furrow_state.json")


class Orchestrator:
    def __init__(self, goal: str, client: LLMClient | None = None, workspace: Path | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.workspace = workspace or Path.cwd()
        self.current_plan: Plan | None = None
        self.task_history: list[TaskModel] = []
        self._load_state()

    def _load_state(self) -> None:
        state_path = self.workspace / STATE_FILE
        if state_path.exists():
            try:
                data = json.loads(state_path.read_text())
                self.cycles = data.get("cycles", 0)
                self.goal = data.get("goal", self.goal)
                self.task_history = [TaskModel(**t) for t in data.get("task_history", [])]
                console.print(f"[dim]Loaded state: {self.cycles} cycles, {len(self.task_history)} tasks in history[/dim]")
                logger.info("state_loaded", cycles=self.cycles, task_count=len(self.task_history))
            except (OSError, json.JSONDecodeError, ValueError) as e:
                console.print(f"[yellow]Could not load state: {e}[/yellow]")
                logger.warning("state_load_failed", error=str(e))

    def _save_state(self) -> None:
        state_path = self.workspace / STATE_FILE
        data = {
            "cycles": self.cycles,
            "goal": self.goal,
            "task_history": [t.model_dump() for t in self.task_history],
        }
        try:
            state_path.write_text(json.dumps(data, indent=2))
            logger.debug("state_saved", cycles=self.cycles, task_count=len(self.task_history))
        except (OSError, ValueError) as e:
            console.print(f"[yellow]Could not save state: {e}[/yellow]")
            logger.warning("state_save_failed", error=str(e))

    async def run(self) -> None:
        logger.info("orchestrator_start", goal=self.goal)
        console.print(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            logger.info("cycle_start", cycle=self.cycles)
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            self._save_state()
            if self._is_done():
                logger.info("orchestrator_complete", cycles=self.cycles)
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal, self.task_history, self.workspace)
        self.current_plan = plan
        logger.info("plan_created", cycle=self.cycles, task_count=len(plan.tasks))
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.info("no_tasks_planned", cycle=self.cycles)
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                self._run_worker(task)
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)
            self.task_history.append(task)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, workspace=self.workspace).run(self.goal, plan.tasks)

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", cycle=self.cycles, summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", cycle=self.cycles, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker(self, task: TaskModel) -> str:
        """Run a single worker task. Override in subclasses for monitoring."""
        logger.info("worker_start", task_id=task.id, description=task.description)
        result = await WorkerAgent(task=task, client=self.client, workspace=self.workspace).run()
        logger.info("worker_end", task_id=task.id, status=task.status)
        return result

    def _is_done(self) -> bool:
        if not self.current_plan:
            return False
        tasks = self.current_plan.tasks
        if not tasks:
            return True
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    def _get_tasks(self) -> list[TaskModel]:
        return self.current_plan.tasks if self.current_plan else []
