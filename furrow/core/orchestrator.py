from __future__ import annotations

import asyncio
from io import StringIO
from typing import Any, Awaitable, Callable

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

OutputCallback = Callable[[str], Awaitable[None]] | None


def _render_plain(renderable: Any) -> str:
    """Render a rich object to plain text for non-terminal consumers (e.g. WebSocket)."""
    buf = StringIO()
    Console(file=buf, force_terminal=False, color_system=None, width=200).print(renderable)
    return buf.getvalue().rstrip("\n")


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: OutputCallback = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.settings: Settings = self.client.settings
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._last_test_result: TestResult | None = None
        self._on_output = on_output

    async def _emit(self, message: Any) -> None:
        """Send a message to the console and optionally to an output callback."""
        console.print(message)
        if self._on_output is not None:
            plain = _render_plain(message)
            if plain:
                await self._on_output(plain)

    async def run(self) -> None:
        await self._emit(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            max_cycles = self.settings.max_cycles
            if max_cycles > 0 and self.cycles >= max_cycles:
                await self._emit("[yellow]Max cycles reached. Halting.[/yellow]")
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
        self._tasks = plan.tasks
        await self._emit(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        if not plan.tasks:
            await self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            worker_tasks = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*worker_tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                await self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                await self._emit(f"[green]Task {task.id} completed[/green]")

        with Status("[bold yellow]Testing...", console=console):
            self._last_test_result = await TesterAgent(client=self.client).run(
                self.goal, plan.tasks
            )

        if self._last_test_result.passed:
            await self._emit(
                f"[green]Tests passed: {self._last_test_result.summary}[/green]"
            )
        else:
            await self._emit(
                f"[red]Tests failed: {self._last_test_result.summary}[/red]"
            )
            for failure in self._last_test_result.failures:
                await self._emit(f"  • {failure}")
            await self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(
                self._last_test_result.failures
            )

    def _is_done(self) -> bool:
        """Check whether the goal is fully complete.

        The goal is done when:
        - There are no remaining tasks AND (no test result exists OR the last
          test result passed), OR
        - All tasks completed without any failures AND tests passed.
        """
        if not self._tasks:
            if self._last_test_result is None:
                return True
            return self._last_test_result.passed

        all_completed = all(t.status == "completed" for t in self._tasks)
        any_failed = any(t.status == "failed" for t in self._tasks)
        tests_passed = (
            self._last_test_result is not None and self._last_test_result.passed
        )
        return all_completed and not any_failed and tests_passed

    def _get_tasks(self) -> list[TaskModel]:
        """Return the tasks from the most recent cycle."""
        return self._tasks
