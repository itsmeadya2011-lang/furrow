from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.config import settings as _default_settings
from furrow.llm import LLMClient

console = Console()
logger = logging.getLogger(__name__)

# Type alias for event callbacks
EventCallback = Callable[[str, Any], Awaitable[None] | None]


class Orchestrator:
    """Runs the infinite development loop: plan → execute → test → repeat."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.settings = settings or _default_settings
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self._tasks: list[TaskModel] = []
        self._on_event = on_event
        self._last_test_result: TestResult | None = None

    # -- Event streaming -------------------------------------------------

    async def _emit(self, event: str, **data: Any) -> None:
        """Notify observers of a lifecycle event."""
        logger.debug("Event: %s %s", event, data)
        if self._on_event is not None:
            try:
                result = self._on_event(event, **data)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:  # noqa: BLE001 - never crash the loop on observer error
                logger.warning("Event callback failed: %s", e)

    # -- Lifecycle -------------------------------------------------------

    async def run(self) -> None:
        await self._emit(
            "start", goal=self.goal, max_cycles=self.settings.max_cycles
        )
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        while True:
            self.cycles += 1
            console.print(f"\n[bold cyan]═══ Cycle {self.cycles} ═══[/bold cyan]")

            # Enforce max_cycles (0 = unlimited)
            if self.settings.max_cycles > 0 and self.cycles > self.settings.max_cycles:
                console.print(
                    f"[yellow]Max cycles ({self.settings.max_cycles}) reached. Stopping.[/yellow]"
                )
                await self._emit("limit_reached", cycles=self.cycles)
                break

            await self._emit("cycle_start", cycle=self.cycles)
            await self._cycle()
            await self._emit(
                "cycle_end",
                cycle=self.cycles,
                tasks=self._tasks,
            )

            if self._is_done():
                console.print("[bold green]Goal complete. Halting.[/bold green]")
                await self._emit("done", cycles=self.cycles)
                break

    async def _cycle(self) -> None:
        # 1. Plan
        plan = await self._plan()
        if plan is None:
            # Planning error — bail out to avoid an infinite loop
            return

        # 2. Execute
        await self._execute(plan)

        # 3. Test
        await self._test()

    # -- Phase: Plan -----------------------------------------------------

    async def _plan(self) -> Optional[Plan]:
        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self.goal)
            except Exception as e:
                console.print(f"[red]Planning failed: {e}[/red]")
                await self._emit("plan_error", error=str(e))
                return None

        console.print(
            Panel(Pretty(plan.model_dump()), title="Plan", border_style="blue")
        )

        # Persist tasks so _is_done() and tests can inspect them
        self._tasks = plan.tasks
        await self._emit("plan", tasks=plan.tasks, rationale=plan.rationale)

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return plan

        # Respect max_parallel_tasks from settings (0 = unlimited)
        max_tasks = self.settings.max_parallel_tasks
        if max_tasks > 0 and len(plan.tasks) > max_tasks:
            console.print(
                f"[yellow]Limiting {len(plan.tasks)} tasks to {max_tasks} parallel.[/yellow]"
            )
            plan.tasks = plan.tasks[:max_tasks]
            self._tasks = plan.tasks

        return plan

    # -- Phase: Execute --------------------------------------------------

    async def _execute(self, plan: Plan) -> None:
        if not plan.tasks:
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            coros = [
                WorkerAgent(task=task, client=self.client).run()
                for task in plan.tasks
            ]
            results = await asyncio.gather(*coros, return_exceptions=True)

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")

        await self._emit("execute", tasks=plan.tasks)

    # -- Phase: Test -----------------------------------------------------

    async def _test(self) -> None:
        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(
                self.goal, self._tasks
            )

        self._last_test_result = test_result

        if test_result.passed:
            console.print(f"[green]Tests passed: {test_result.summary}[/green]")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            # Refine the goal for the next cycle
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        await self._emit("test", result=test_result)

    # -- Helpers ---------------------------------------------------------

    def _is_done(self) -> bool:
        """Return True when the goal is considered complete.

        Done means: tasks were planned, all completed, none failed, and
        tests passed.  If no tasks are planned, the goal is trivially done.
        """
        if not self._tasks:
            return True

        failed = sum(1 for t in self._tasks if t.status == "failed")
        completed = sum(1 for t in self._tasks if t.status == "completed")

        if failed > 0:
            return False
        if completed < len(self._tasks):
            return False

        # Tasks are done — confirm with test result
        if self._last_test_result is not None:
            return self._last_test_result.passed

        return True

    def _get_tasks(self) -> list[Any]:
        """Return the current cycle's tasks (persisted between phases)."""
        return self._tasks
