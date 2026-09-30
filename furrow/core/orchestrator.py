from __future__ import annotations

import asyncio
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient
from furrow.logging import logger

console = Console()


class Orchestrator:
    """Orchestrates the infinite development loop: plan → execute → test → repeat."""

    def __init__(self, goal: str, client: LLMClient | None = None) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None

    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )
        log = logger.bind(goal=self.goal)
        while True:
            if (
                self.client.settings.max_cycles > 0
                and self.cycles >= self.client.settings.max_cycles
            ):
                console.print("[yellow]Max cycles reached. Stopping.[/yellow]")
                log.info("max_cycles_reached", cycles=self.cycles)
                break
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                log.info("goal_complete", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Store tasks for _is_done() and _get_tasks()
        self.tasks = plan.tasks

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await self._execute_tasks(plan.tasks)

        for task, result in zip(plan.tasks, results, strict=False):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self.last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _execute_tasks(self, tasks: list[TaskModel]) -> list[Any]:
        """Execute tasks in parallel, limited by max_parallel_tasks."""
        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_worker(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        workers = [run_worker(task) for task in tasks]
        return await asyncio.gather(*workers, return_exceptions=True)

    def _is_done(self) -> bool:
        """Determine if the goal is complete.

        Returns True when all planned tasks are completed AND the last test
        run passed. Returns False if any task failed or if tests are still
        failing (so the next cycle can attempt fixes).
        """
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = sum(1 for t in tasks if t.status == "failed")
        completed = sum(1 for t in tasks if t.status == "completed")
        if failed > 0 and completed == 0:
            # All tasks failed — stop to avoid infinite retry on same plan
            return True
        if failed > 0:
            return False
        if completed < len(tasks):
            return False
        # All tasks completed — check test results
        return self.last_test_result is None or self.last_test_result.passed

    def _get_tasks(self) -> list[TaskModel]:
        """Return tasks from the most recent planning cycle."""
        return self.tasks

    @property
    def status(self) -> str:
        """Human-readable status of the orchestrator."""
        if not self.tasks:
            return "idle"
        completed = sum(1 for t in self.tasks if t.status == "completed")
        failed = sum(1 for t in self.tasks if t.status == "failed")
        total = len(self.tasks)
        return f"{completed}/{total} completed, {failed} failed, cycle {self.cycles}"
