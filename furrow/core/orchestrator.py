from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.all_tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None
        self.max_cycles: int = self.client.settings.max_cycles
        self.output_callback = output_callback

    async def _log(self, message: str) -> None:
        """Print to console and optionally stream to output callback."""
        console.print(message)
        if self.output_callback is not None:
            await self.output_callback(message)

    async def run(self) -> None:
        await self._log(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            # Check max_cycles limit
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                await self._log(
                    f"[yellow]Reached max_cycles ({self.max_cycles}). Stopping.[/yellow]"
                )
                break

            self.cycles += 1
            await self._log(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                await self._log("[bold green]Goal complete. Halting.[/bold green]")
                break

            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                await self._log(
                    f"[yellow]Reached max_cycles ({self.max_cycles}). Stopping.[/yellow]"
                )
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            plan = await self.planner.plan(self.goal)
        await self._log(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._log("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        # Store all tasks for tracking across cycles
        self.all_tasks.extend(plan.tasks)

        # Execute tasks in parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> str:
            async with semaphore:
                return await WorkerAgent(task=task, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [_run_task(task) for task in plan.tasks]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._log(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._log(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            self.last_test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        test_result = self.last_test_result
        if test_result.passed:
            await self._log(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._log(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._log(f"  • {failure}")
            await self._log("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return False

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")

        # If all tasks completed successfully, we're done
        if completed == len(tasks) and failed == 0:
            # Also require tests to have passed
            if self.last_test_result and self.last_test_result.passed:
                return True

        return False

    def _get_tasks(self) -> list[TaskModel]:
        return self.all_tasks
