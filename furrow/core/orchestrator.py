from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, TestResult
from furrow.core.state import State, StateManager
from furrow.llm import LLMClient

logger = structlog.get_logger(__name__)

console = Console()

LogListener = Callable[["LogEvent"], Awaitable[None]]
_COLORS = {"info": "green", "warning": "yellow", "error": "red", "step": "cyan"}


@dataclass
class LogEvent:
    """A structured event emitted during the run loop."""

    level: str = "info"  # info, warning, error, step
    message: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


class OrchestratorStop(Exception):
    """Raised (e.g. by a listener) to request a clean stop of the loop."""


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        state: State | None = None,
        listeners: list[LogListener] | None = None,
        persist: bool = False,
    ) -> None:
        self.goal = goal
        self.settings = settings or Settings()
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client)
        self.cycles = 0
        self.logger = logger.bind(goal=goal)
        self.listeners: list[LogListener] = list(listeners or [])
        self.persist = persist
        self._tasks: list[TaskModel] = list(state.tasks) if state else []
        self._last_test_result: TestResult | None = state.test_result if state else None
        self.state = state or State(goal=goal)
        self.state_manager = StateManager(
            self.settings.state_file or StateManager.default_path()
        )

    # -- event emission -------------------------------------------------
    async def emit(self, event: LogEvent) -> None:
        """Render an event to the console (rich) and dispatch to listeners."""
        self.logger.debug(event.level, message=event.message, **event.meta)
        color = _COLORS.get(event.level, "")
        if color:
            console.print(f"[{color}]{event.message}[/{color}]")
        else:
            console.print(event.message)
        for listener in self.listeners:
            try:
                await listener(event)
            except OrchestratorStop:
                raise
            except Exception as e:
                self.logger.warning("listener_error", error=str(e), exc_info=True)

    async def _log(self, message: str, level: str = "info", **meta: Any) -> None:
        await self.emit(LogEvent(level=level, message=message, meta=meta))

    # -- main loop ------------------------------------------------------
    async def run(self) -> None:
        console.print(
            Panel.fit(f"[bold green]Furrow[/bold green]\nGoal: {self.goal}", title="Furrow")
        )

        if self.persist:
            loaded = await self.state_manager.load()
            if loaded is not None and loaded.goal == self.goal:
                self.state = loaded
                self.cycles = loaded.cycle
                self._tasks = list(loaded.tasks)
                self._last_test_result = loaded.test_result
                await self._log(
                    f"Resumed from saved state: cycle {loaded.cycle}, "
                    f"{len(loaded.tasks)} tasks tracked.",
                    level="info",
                )

        try:
            while True:
                self.cycles += 1
                self.state.cycle = self.cycles
                if self.settings.max_cycles and self.cycles > self.settings.max_cycles:
                    await self._log(
                        f"Reached max_cycles ({self.settings.max_cycles}). Stopping.",
                        level="info",
                    )
                    break
                await self._cycle()
                if self._is_done():
                    await self._log("Goal complete. Halting.", level="info")
                    break
        except OrchestratorStop:
            await self._log("Orchestration stopped by listener.", level="info")
        finally:
            if self.persist:
                await self.state_manager.save(self.state)

    async def _cycle(self) -> None:
        with Status("[bold yellow]Planning...", console=console):
            plan = await self.planner.plan(self.goal)
        self._tasks = plan.tasks
        self.state.tasks = plan.tasks
        self.state.goal = self.goal
        console.print(
            Panel(
                Pretty(plan.model_dump()),
                title=f"Cycle {self.cycles} Plan",
                border_style="blue",
            )
        )
        await self._log(f"Plan: {len(plan.tasks)} task(s).", level="step", tasks=len(plan.tasks))

        if not plan.tasks:
            console.print("[yellow]No tasks planned. Goal may be complete.[/yellow]")
            return

        with Status("[bold yellow]Executing tasks in parallel...", console=console):
            semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)

            async def _run_task(task: TaskModel) -> Any:
                async with semaphore:
                    return await WorkerAgent(task=task, client=self.client).run()

            results = await asyncio.gather(
                *[_run_task(t) for t in plan.tasks], return_exceptions=True
            )

        for task, result in zip(plan.tasks, results):
            if isinstance(result, Exception):
                task.status = "failed"
                task.result = str(result)
                console.print(f"[red]Task {task.id} failed: {result}[/red]")
                await self._log(f"Task {task.id} failed: {result}", level="error")
            else:
                task.status = "completed"
                task.result = result
                console.print(f"[green]Task {task.id} completed[/green]")
                await self._log(f"Task {task.id} completed", level="info")

        self.state.tasks = plan.tasks

        with Status("[bold yellow]Testing...", console=console):
            test_result = await TesterAgent(client=self.client).run(self.goal, plan.tasks)

        self._last_test_result = test_result
        self.state.test_result = test_result
        if test_result.passed:
            await self._log(f"Tests passed: {test_result.summary}", level="info")
        else:
            console.print(f"[red]Tests failed: {test_result.summary}[/red]")
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            await self._log(f"Tests failed: {test_result.summary}", level="warning")
            await self._log("Will attempt fix in next cycle.", level="info")
            self.goal = "Fix failing tests:\n" + "\n".join(test_result.failures)

        if self.persist:
            await self.state_manager.save(self.state)

    # -- completion -----------------------------------------------------
    def _is_done(self) -> bool:
        """A cycle is done when every task completed and tests pass (or none)."""
        tasks = self._get_tasks()
        if not tasks:
            return self._last_test_result is None or self._last_test_result.passed

        completed = sum(1 for t in tasks if t.status == "completed")
        failed = sum(1 for t in tasks if t.status == "failed")
        if failed > 0:
            return False
        if completed < len(tasks):
            return False
        # All tasks completed; still not done if tests failed.
        return self._last_test_result is None or self._last_test_result.passed

    def _get_tasks(self) -> list[Any]:
        """Return the task list from the most recent cycle."""
        return self._tasks
