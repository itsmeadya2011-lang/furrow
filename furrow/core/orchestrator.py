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
from furrow.config import Plan
from furrow.llm import LLMClient

console = Console()

MAX_RETRIES = 3


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
        self.current_plan: Plan | None = None
        self.output_callback = output_callback
        max_parallel = self.client.settings.max_parallel_tasks
        self._semaphore = asyncio.Semaphore(max(1, max_parallel))

    def _emit(self, msg: Any) -> None:
        """Print a message to the console and, if configured, to the callback.

        Rich renderables (Panel, Pretty, etc.) are rendered to plain text
        before being passed to the callback so that WebSocket clients receive
        clean text instead of rich objects or markup tags.
        """
        console.print(msg)
        if self.output_callback is not None:
            with console.capture() as capture:
                console.print(msg, end="")
            self.output_callback(capture.get())

    def _get_tasks(self) -> list:
        """Return the tasks for the current plan (empty when no plan has run)."""
        if self.current_plan is None:
            return []
        return self.current_plan.tasks

    def _is_done(self) -> bool:
        tasks = self._get_tasks()
        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        return completed >= len(tasks)

    async def run(self, output_callback: Callable[[str], None] | None = None) -> None:
        if output_callback is not None:
            self.output_callback = output_callback

        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        max_cycles = self.client.settings.max_cycles
        while True:
            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()
            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break
            if max_cycles > 0 and self.cycles >= max_cycles:
                self._emit("[bold yellow]Reached max_cycles. Halting.[/bold yellow]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self.current_plan = plan
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            results = await asyncio.gather(
                *(self._run_worker(task) for task in plan.tasks),
                return_exceptions=True,
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, BaseException):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        if test_result.passed:
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    async def _run_worker(self, task) -> str:
        """Run a worker for a task, retrying transient failures up to MAX_RETRIES times."""
        worker = WorkerAgent(task=task, client=self.client)
        last_err: Exception | None = None
        for attempt in range(1, MAX_RETRIES + 1):
            async with self._semaphore:
                try:
                    return await worker.run()
                except Exception as e:  # noqa: BLE001 — retry loop catches all worker errors
                    last_err = e
                    self._emit(f"[red]Task {task.id} attempt {attempt}/{MAX_RETRIES} failed: {e}[/red]")
        assert last_err is not None
        raise last_err
