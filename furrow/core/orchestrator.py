from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient

console = Console()

OutputCallback = Callable[[str], None] | Callable[[str], Awaitable[None]]


class Orchestrator:
    """Runs the planning → execution → testing loop until the goal is complete.

    Parameters
    ----------
    goal:
        The high-level task description.
    client:
        Optional shared :class:`LLMClient`. If omitted a new one is created.
    on_output:
        Optional callback (sync or async) that receives plain-text progress
        messages.  The CLI uses Rich console output; the web server passes a
        callback that streams to the WebSocket client.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_output: OutputCallback | None = None,
    ) -> None:
        self.goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.tester = TesterAgent(client=self.client)
        self.cycles = 0

        # Cross-cycle state
        self._task_history: list[TaskModel] = []
        self._cycle_total_count: int = 0
        self._cycle_failed_count: int = 0
        self._test_passed: bool = False
        self._last_plan: Plan | None = None

        # Output callback
        self._on_output: OutputCallback | None = on_output
        self.max_cycles: int = self.client.settings.max_cycles

    # ------------------------------------------------------------------ #
    #  Output helpers
    # ------------------------------------------------------------------ #

    def _emit(self, msg: str | Any) -> None:
        """Emit a plain-text or Rich-renderable message to console + callback.

        Used for Rich-only output (e.g. Panel display) where we don't need
        to await an async callback.
        """
        console.print(msg)

    @staticmethod
    def _strip_rich_markup(text: str) -> str:
        """Remove Rich markup tags like [bold], [red]text[/red], etc."""
        from rich.text import Text

        return Text.from_markup(text).plain

    @staticmethod
    def _to_plain_text(renderable: Any) -> str:
        """Render a Rich renderable (Panel, Pretty, etc.) to plain text."""
        capture_console = Console(
            force_terminal=False,
            no_color=True,
            width=console.width,
            highlight=False,
        )
        with capture_console.capture() as cap:
            capture_console.print(renderable)
        return cap.get()

    async def _aemit(self, msg: str | Any) -> None:
        """Async-friendly emit that properly awaits async callbacks."""
        console.print(msg)
        if self._on_output is not None:
            text_msg = self._to_plain_text(msg) if not isinstance(msg, str) else msg
            clean = self._strip_rich_markup(text_msg)
            result = self._on_output(clean)
            if asyncio.iscoroutine(result):
                await result

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    async def run(self) -> None:
        await self._aemit("[bold green]Furrow[/bold green]\nGoal: " + self.goal)
        while True:
            # Enforce max_cycles (0 = infinite) — check *before* incrementing
            if self.max_cycles > 0 and self.cycles >= self.max_cycles:
                await self._aemit("[yellow]Max cycles reached. Stopping.[/yellow]")
                break

            self.cycles += 1
            await self._aemit(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")
            await self._cycle()

            if self._is_done():
                await self._aemit("[bold green]Goal complete. Halting.[/bold green]")
                break

    # ------------------------------------------------------------------ #
    #  Single cycle
    # ------------------------------------------------------------------ #

    async def _cycle(self) -> None:
        # 1. Plan ---------------------------------------------------------
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._last_plan = plan
        self._emit(Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue"))

        if not plan.tasks:
            await self._aemit("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            self._cycle_total_count = 0
            self._cycle_failed_count = 0
            self._test_passed = True
            return

        # 2. Execute tasks (dependency-aware tiers, parallel within tier) --
        self._cycle_total_count = len(plan.tasks)
        self._cycle_failed_count = 0

        semaphore = asyncio.Semaphore(self.client.settings.max_parallel_tasks)

        async def run_task(task: TaskModel) -> str:
            async with semaphore:
                worker = WorkerAgent(task=task, client=self.client)
                return await worker.run()

        for tier in self._topological_tiers(plan.tasks):
            results = await asyncio.gather(
                *[run_task(t) for t in tier],
                return_exceptions=True,
            )
            for task, result in zip(tier, results):
                self._task_history.append(task)
                if isinstance(result, BaseException):
                    task.status = "failed"
                    task.result = str(result)
                    self._cycle_failed_count += 1
                    await self._aemit(f"[red]Task {task.id} failed: {result}[/red]")
                else:
                    task.status = "completed"
                    task.result = result
                    await self._aemit(f"[green]Task {task.id} completed[/green]")

        # 3. Test ---------------------------------------------------------
        with Status("[bold yellow]Testing...", console=console):
            test_result: TestResult = await self.tester.run(self.goal, plan.tasks)

        self._test_passed = test_result.passed
        if test_result.passed:
            await self._aemit(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            await self._aemit(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                await self._aemit(f"  • {failure}")
            await self._aemit("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

    # ------------------------------------------------------------------ #
    #  Dependency ordering
    # ------------------------------------------------------------------ #

    @staticmethod
    def _topological_tiers(tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Group tasks into tiers so that dependencies are executed first.

        Tier 0 contains tasks with no dependencies.  Tier *n* contains tasks
        all of whose dependencies appear in tiers 0..n-1.  Within a tier,
        tasks are executed in parallel.
        """
        remaining = {t.id: t for t in tasks}
        completed_ids: set[str] = set()
        tiers: list[list[TaskModel]] = []

        while remaining:
            tier = [
                t for t in remaining.values()
                if all(dep in completed_ids for dep in t.dependencies)
            ]
            if not tier:
                # Circular dependency — break it by taking all remaining.
                tier = list(remaining.values())
            for t in tier:
                del remaining[t.id]
            completed_ids.update(t.id for t in tier)
            tiers.append(tier)

        return tiers

    # ------------------------------------------------------------------ #
    #  Completion check
    # ------------------------------------------------------------------ #

    def _is_done(self) -> bool:
        # No tasks were planned → goal is (already) complete
        if self._cycle_total_count == 0:
            return True
        # All tasks in the last cycle passed and tests succeeded;
        # otherwise retry to fix
        return self._cycle_failed_count == 0 and self._test_passed

    def _get_tasks(self) -> list[TaskModel]:
        """Return the full history of tasks across all cycles."""
        return self._task_history
