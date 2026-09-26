from __future__ import annotations

import asyncio
from typing import cast

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status
from structlog import get_logger

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()
logger = get_logger(__name__)


class Orchestrator:
    """Runs the infinite Furrow development loop: plan -> execute -> test -> repeat."""

    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        logger.info("orchestrator_start", goal=self.goal)
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            logger.info("cycle_start", cycle=self.cycles)

            await self._cycle()

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                logger.info("orchestrator_complete", cycles=self.cycles)
                break

            max_cycles = self.client.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                console.print(
                    f"[yellow]Max cycles ({max_cycles}) reached. Stopping.[/yellow]"
                )
                logger.warning("max_cycles_reached", max_cycles=max_cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            logger.warning("no_tasks_planned", goal=self.goal)
            return

        # Persist tasks across cycles so _is_done() and status reporting work.
        self.all_tasks.extend(plan.tasks)

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results, strict=True):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                logger.error("task_failed", task_id=task.id, error=str(result))
            else:
                task.status = "completed"
                task.result = cast("str", result)
                console.print(f"[green]Task {task.id} completed[/green]")
                logger.info("task_completed", task_id=task.id)

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
            logger.info("tests_passed", summary=test_result.summary)
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            logger.warning("tests_failed", summary=test_result.summary, failures=test_result.failures)
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        """Return True when the most recent test run passed (or no tests were needed)."""
        if self.last_test_result is None:
            return False
        return self.last_test_result.passed

    @property
    def completed_tasks(self) -> list[TaskModel]:
        return [t for t in self.all_tasks if t.status == "completed"]

    @property
    def failed_tasks(self) -> list[TaskModel]:
        return [t for t in self.all_tasks if t.status == "failed"]
