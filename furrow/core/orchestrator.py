from __future__ import annotations

import asyncio
import structlog
from typing import Awaitable, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, settings
from furrow.llm import LLMClient

log = structlog.get_logger(__name__)

console = Console()

OutputCallback = Callable[[str], Awaitable[None]] | Callable[[str], None]


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings_obj: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings_obj or settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._current_plan: Optional[Plan] = None
        self._output_callback: OutputCallback | None = None
        self._stop_requested = False
        self._tests_passed: bool = False

    def set_output_callback(self, callback: OutputCallback | None = None) -> None:
        """Set a callback that receives output strings (for web streaming)."""
        self._output_callback = callback

    def request_stop(self) -> None:
        """Request the orchestrator to stop after the current cycle."""
        self._stop_requested = True

    def _emit(self, msg: str) -> None:
        """Send output to console and optionally to the output callback.

        For the callback, the message is rendered to plain text so that
        rich markup and renderables (Panel, Pretty) are converted to
        readable strings suitable for web streaming.
        """
        log.info("orchestrator_output", message=str(msg))
        if self._output_callback is not None:
            with console.capture() as capture:
                console.print(msg)
            plain_text = capture.get().rstrip()
            if plain_text:
                try:
                    result = self._output_callback(plain_text)
                    if asyncio.iscoroutine(result):
                        asyncio.create_task(result)
                except Exception as e:
                    log.warning("output_callback_error", error=str(e))
        console.print(msg)

    async def run(self) -> None:
        self._emit(Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow"))
        while True:
            if self._stop_requested:
                self._emit("[yellow]Stop requested. Halting.[/yellow]")
                break

            # Enforce max_cycles (0 means unlimited)
            if self.settings.max_cycles > 0 and self.cycles >= self.settings.max_cycles:
                self._emit(f"[yellow]Reached max_cycles ({self.settings.max_cycles}). Stopping.[/yellow]")
                break

            self.cycles += 1
            self._emit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            try:
                await self._cycle()
            except Exception as e:
                log.error("cycle_error", error=str(e), exc_info=True)
                self._emit(f"[red]Cycle {self.cycles} error: {e}[/red]")

            if self._is_done():
                self._emit("[bold green]Goal complete. Halting.[/bold green]")
                break

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console) as status:
            try:
                plan = await self.planner.plan(self.goal)
            except ValueError as e:
                self._emit(f"[red]Planning failed: {e}[/red]")
                # No plan — nothing to do
                return

        self._current_plan = plan
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        # Update tracked tasks with the new plan's tasks
        for task in plan.tasks:
            # Preserve status if we've seen this task before
            existing = next((t for t in self._tasks if t.id == task.id), None)
            if existing:
                task.status = existing.status
                task.result = existing.result
            else:
                self._tasks.append(task)

        if not plan.tasks:
            self._emit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            tasks = [
                WorkerAgent(task=task, client=self.client, settings=self.settings).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                self._emit(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                self._emit(f"[green]Task {task.id} completed[/green]")

        # Update tracked tasks with results
        for task in plan.tasks:
            existing = next((t for t in self._tasks if t.id == task.id), None)
            if existing:
                existing.status = task.status
                existing.result = task.result

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client, settings=self.settings).run(
                self.goal, plan.tasks
            )

        if test_result.passed:
            self._tests_passed = True
            self._emit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            self._tests_passed = False
            self._emit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                self._emit(f"  • {failure}")
            self._emit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = self.goal + "\n\nFix failing tests from previous cycle:\n" + "\n".join(
                test_result.failures
            )

    def _is_done(self) -> bool:
        """Check if all tasks are completed and tests passed across cycles."""
        tasks = self._get_tasks()
        if not tasks:
            # No tasks planned — nothing left to do
            return True
        all_completed = all(t.status == "completed" for t in tasks)
        any_pending = any(t.status == "pending" for t in tasks)
        any_failed = any(t.status == "failed" for t in tasks)

        # Done only when all tasks have been attempted, none pending or failed,
        # AND tests passed in the most recent cycle
        if any_pending:
            return False
        if any_failed:
            return False
        if not all_completed:
            return False
        return self._tests_passed

    def _get_tasks(self) -> list[TaskModel]:
        """Return all tracked tasks across all cycles."""
        return self._tasks
