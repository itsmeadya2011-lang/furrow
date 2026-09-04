from __future__ import annotations

import asyncio
from typing import Any, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TestResult
from furrow.llm import LLMClient

console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        output_callback: Callable[[str], None] | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[Any] = []
        self._last_test_result: TestResult | None = None
        self.output_callback = output_callback

    def _emit(self, text: str) -> None:
        """Send a status/log message to the output callback or console."""
        if self.output_callback is not None:
            self.output_callback(text)
        else:
            console.print(text)

    async def run(self) -> None:
        max_cycles = self.client.settings.max_cycles
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            self.cycles += 1
            if max_cycles > 0 and self.cycles > max_cycles:
                self._emit("[yellow]Max cycles reached. Stopping.[/yellow]")
                break
            self._emit(f"\n[bold cyan]== Cycle {self.cycles} ==[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        console.print(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        self._tasks = plan.tasks
        self._emit(f"Planned {len(self._tasks)} task(s).")

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def _bounded(task_model: Any) -> str:
            async with semaphore:
                return await WorkerAgent(task=task_model, client=self.client).run()

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(_bounded(t) for t in plan.tasks),
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            self._last_test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if self._last_test_result.passed:
            self._emit(f"[green]Tests passed: {self._last_test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {self._last_test_result.summary}[/red]")
            for failure in self._last_test_result.failures:
                self._emit(f"  - {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(self._last_test_result.failures)

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        if not tasks:
            return True
        failed = [t for t in tasks if t.status == "failed"]
        if failed:
            return False
        completed = [t for t in tasks if t.status == "completed"]
        if len(completed) < len(tasks):
            return False
        # All tasks completed - only done if the last test run passed
        if self._last_test_result is not None and not self._last_test_result.passed:
            return False
        return True

    def _get_tasks(self) -> list[Any]:
        return self._tasks