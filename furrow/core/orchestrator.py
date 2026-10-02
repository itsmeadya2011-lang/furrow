from __future__ import annotations

import asyncio
from collections.abc import Callable
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

console = Console()


# Type for the optional output callback (e.g. WebSocket streaming)
MessageHandler = Callable[[str], Any]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_message: MessageHandler | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._test_results: list[TestResult] = []
        self._on_message = on_message

    async def _emit(self, message) -> None:
        """Print to console and optionally forward to a streaming callback."""
        console.print(message)
        if self._on_message:
            if isinstance(message, str):
                msg = message
            else:
                with console.capture() as capture:
                    console.print(message)
                msg = capture.get()
            result = self._on_message(msg)
            if asyncio.iscoroutine(result):
                await result

    async def run(self) -> None:
        await self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self.client.settings.max_cycles > 0 and self.cycles >= self.client.settings.max_cycles:
                await self._emit(
                    f"[yellow]Reached max_cycles ({self.client.settings.max_cycles}). Stopping.[/yellow]"
                )
                break
            self.cycles += 1
            await self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                await self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        await self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._tasks = []
            return

        self._tasks = plan.tasks

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _run_task(task: TaskModel) -> tuple[TaskModel, str | BaseException]:
            async with semaphore:
                try:
                    result = await WorkerAgent(task=task, client=self.client).run()
                    return task, result
                except Exception as e:  # noqa: BLE001 - intentionally broad to capture any worker failure
                    return task, e

        coros = [_run_task(task) for task in plan.tasks]
        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in results:
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)
        self._test_results.append(test_result)

        if test_result.passed:
            await self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    def _is_done(self) -> bool:
        if not self._tasks:
            return False
        completed = sum(1 for t in self._tasks if t.status == "completed")
        failed = sum(1 for t in self._tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed >= len(self._tasks):
            # All tasks completed — check that the last test run passed
            return not self._test_results or self._test_results[-1].passed
        return False

    def _get_tasks(self) -> list[Any]:
        return self._tasks
