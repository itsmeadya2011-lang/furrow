"""The Furrow orchestration loop.

The orchestrator owns the plan -> execute -> test cycle. It keeps real state
between cycles (tasks, cycle counter, test results) so the loop can actually
loop: each cycle asks the planner for the next slice of work, runs the workers
in parallel under a concurrency cap, runs the test suite, and feeds failures
back to the planner on the next pass -- all without ever mutating the original
goal.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, TypeVar

from rich.console import Console
from rich.panel import Panel
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import CycleReport, Plan, RunResult, TaskModel, TaskStatus, TestResult, settings
from furrow.llm import LLMClient

console = Console()

#: Diagnostics only. The orchestrator never configures handlers or the root logger.
logger = logging.getLogger(__name__)

#: Async progress sink. The web server passes a coroutine that awaits a websocket send.
ProgressCallback = Callable[[str], Awaitable[None]]

#: Task states that will not change again during a run.
SETTLED_STATUSES = frozenset({TaskStatus.COMPLETED, TaskStatus.SKIPPED})

#: The only values ``RunResult.stopped_reason`` may take.
GOAL_COMPLETE = "goal_complete"
MAX_CYCLES = "max_cycles"
STOPPED = "stopped"
ERROR = "error"

_HALT_LABELS = {
    GOAL_COMPLETE: "Goal complete",
    MAX_CYCLES: "Cycle limit reached",
    STOPPED: "Stopped by request",
    ERROR: "Stopped after error",
}

#: Longest worker/test excerpt echoed to the console.
_PREVIEW_LIMIT = 120
#: Most test failures listed for a single cycle before truncating.
_MAX_LISTED_FAILURES = 5

_T = TypeVar("_T")


class Orchestrator:
    """Runs the plan/execute/test loop until the goal is met, capped, or stopped."""

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        *,
        on_progress: ProgressCallback | None = None,
        max_cycles: int | None = None,
    ) -> None:
        self._goal = goal
        self.client = client or LLMClient()
        self.planner = PlannerAgent(client=self.client)
        self.on_progress: ProgressCallback | None = on_progress
        # ``None`` defers to settings.max_cycles (0 == unlimited).
        self._max_cycles = max_cycles
        self._tasks: list[TaskModel] = []
        self._index: dict[str, int] = {}
        self._cycles = 0
        self._test_results: list[TestResult] = []
        self._reports: list[CycleReport] = []
        self._failures: list[str] = []
        self._tests_passed: bool | None = None
        self._error: str | None = None
        self._plan_was_empty = False
        self._planned_new_work = False
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event = asyncio.Event()

    # ------------------------------------------------------------------ state

    @property
    def tasks(self) -> list[TaskModel]:
        """Every task seen so far, across all cycles."""
        return list(self._tasks)

    @property
    def cycles(self) -> int:
        """Number of cycles run so far."""
        return self._cycles

    @property
    def goal(self) -> str:
        """The original goal. Never overwritten, even when tests fail."""
        return self._goal

    @property
    def stopped(self) -> bool:
        """Whether ``stop()`` has been requested."""
        return self._stop_event.is_set()

    @property
    def _cycle_cap(self) -> int:
        cap = self._max_cycles if self._max_cycles is not None else settings.max_cycles
        return max(0, cap)

    # ------------------------------------------------------------- lifecycle

    async def run(self) -> RunResult:
        """Drive the loop until a halt condition is met, then report the outcome."""
        if self._running:
            raise RuntimeError("Orchestrator.run() is already running")
        self._running = True
        self._reset()
        self._loop = asyncio.get_running_loop()
        try:
            await self._emit(
                f"Goal: {self._goal}",
                renderable=Panel.fit(
                    f"[bold green]Furrow[/bold green]\nGoal: {self._goal}", title="Furrow"
                ),
            )
            while True:
                reason = self._done_reason()
                if reason is not None:
                    return await self._finish(reason)
                self._cycles += 1
                await self._emit(f"═══ Cycle {self._cycles} ═══", style="bold cyan")
                report = await self._cycle()
                if report is not None:
                    self._reports.append(report)
                if self._error is not None:
                    # A planner/tester blowup is recorded, not raised at the caller.
                    return await self._finish(ERROR)
        finally:
            self._running = False
            self._loop = None

    async def stop(self) -> None:
        """Ask the loop to halt. Safe to call from another task or thread."""
        loop = self._loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - stop() is a coroutine
            running = None
        if loop is not None and running is not None and loop is not running:
            # Called from a foreign event loop (another thread): hop back to ours
            # so the waiters are woken on the loop that owns them.
            loop.call_soon_threadsafe(self._stop_event.set)
            return
        self._stop_event.set()

    def _reset(self) -> None:
        """Clear per-run state so ``run()`` can be called again."""
        self._tasks = []
        self._index = {}
        self._cycles = 0
        self._test_results = []
        self._reports = []
        self._failures = []
        self._tests_passed = None
        self._error = None
        self._plan_was_empty = False
        self._planned_new_work = False
        self._stop_event = asyncio.Event()

    # -------------------------------------------------------------- progress

    async def _emit(
        self,
        message: str,
        *,
        style: str | None = None,
        renderable: Any | None = None,
    ) -> None:
        """Print a progress line and forward its plain text to ``on_progress``.

        Every console line in this module goes through here so the web server
        sees exactly what the terminal shows. A consumer that raises (a closed
        websocket, say) must never take the loop down with it.
        """
        if renderable is not None:
            console.print(renderable)
        else:
            console.print(f"[{style}]{message}[/{style}]" if style else message)
        callback = self.on_progress
        if callback is None:
            return
        try:
            await callback(message)
        except Exception as exc:  # noqa: BLE001 - a broken sink is not a loop failure
            logger.debug("progress callback failed: %s: %s", type(exc).__name__, exc)

    @staticmethod
    def _spinner(message: str) -> Status:
        return Status(f"[bold yellow]{message}[/bold yellow]", console=console)

    # ------------------------------------------------------------- halt logic

    def _done_reason(self) -> str | None:
        """Why the loop should halt, or ``None`` to run another cycle."""
        if self._stop_event.is_set():
            return STOPPED
        if self._plan_was_empty:
            # The planner prompt defines an empty task list as "goal complete".
            return GOAL_COMPLETE
        if self._all_settled() and not self._work_remaining():
            return GOAL_COMPLETE
        cap = self._cycle_cap
        if cap > 0 and self._cycles >= cap:
            return MAX_CYCLES
        return None

    def _all_settled(self) -> bool:
        """True when every tracked task is COMPLETED/SKIPPED and none FAILED.

        An empty task list is deliberately *not* settled: "nothing ran yet" is
        not "nothing left to do", which is what made the loop halt after one cycle.
        """
        if not self._tasks:
            return False
        if any(task.status is TaskStatus.FAILED for task in self._tasks):
            return False
        return all(task.status in SETTLED_STATUSES for task in self._tasks)

    def _work_remaining(self) -> bool:
        """Whether another planning pass could still make progress.

        A red test suite keeps the loop alive so the planner gets a chance to
        propose fixes -- but only when the last cycle actually had new work to
        propose, otherwise a planner stuck on an already-finished task would
        spin forever.
        """
        return self._tests_passed is False and self._planned_new_work

    async def _finish(self, reason: str) -> RunResult:
        completed = sum(1 for task in self._tasks if task.status is TaskStatus.COMPLETED)
        failed = sum(1 for task in self._tasks if task.status is TaskStatus.FAILED)
        label = _HALT_LABELS.get(reason, reason)
        await self._emit(
            f"{label} after {self._cycles} cycle(s); "
            f"{completed}/{len(self._tasks)} task(s) completed, {failed} failed.",
            style="bold green" if reason == GOAL_COMPLETE else "bold yellow",
        )
        if reason == ERROR and self._error:
            await self._emit(f"Last error: {self._error}", style="bold red")
        return RunResult(
            goal=self._goal,
            cycles=self._cycles,
            stopped_reason=reason,
            tasks=list(self._tasks),
            test_results=list(self._test_results),
            reports=list(self._reports),
        )

    # ------------------------------------------------------------ one cycle

    async def _cycle(self) -> CycleReport | None:
        """Plan, execute and test once. Returns the report, or ``None`` on error."""
        report = CycleReport(cycle=self._cycles)

        try:
            with self._spinner("Planning..."):
                plan = await self.planner.plan(self._planner_input())
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            self._error = f"{type(exc).__name__}: {exc}"
            report.summary = f"planning failed: {self._error}"
            await self._emit(f"Planning failed: {self._error}", style="bold red")
            return None

        cycle_tasks = self._merge_plan(plan)
        self._plan_was_empty = not cycle_tasks
        report.planned = len(cycle_tasks)
        await self._emit_plan(plan, cycle_tasks)

        if not cycle_tasks:
            report.summary = "planner returned no tasks: goal already complete"
            return report

        runnable = [task for task in cycle_tasks if task.status not in SETTLED_STATUSES]
        self._planned_new_work = bool(runnable)
        for task in cycle_tasks:
            if task.status in SETTLED_STATUSES:
                await self._emit(f"· {task.id} already {task.status.value}", style="dim")
        if runnable:
            await self._emit(
                f"Executing {len(runnable)} task(s) "
                f"(max {max(1, settings.max_parallel_tasks)} in parallel)",
                style="cyan",
            )
            with self._spinner(f"Executing {len(runnable)} task(s)..."):
                await self._execute(runnable, report)
        else:
            report.summary = "planner proposed no new work"

        try:
            with self._spinner("Running tests..."):
                test_result = await self._race_stop(
                    TesterAgent(client=self.client).run(self._goal, cycle_tasks)
                )
        except Exception as exc:  # noqa: BLE001 - reported, never propagated
            self._error = f"{type(exc).__name__}: {exc}"
            report.summary = f"testing failed: {self._error}"
            await self._emit(f"Testing failed: {self._error}", style="bold red")
            return None

        if test_result is None:
            report.summary = "stopped while running tests"
            return report

        self._test_results.append(test_result)
        self._tests_passed = test_result.passed
        self._failures = list(test_result.failures)
        report.tests_passed = test_result.passed
        if test_result.passed:
            report.summary = "tests passed"
            await self._emit(f"Tests passed: {_preview(test_result.summary)}", style="bold green")
        else:
            report.summary = f"{len(test_result.failures)} failing test(s)"
            await self._emit(f"Tests failed: {_preview(test_result.summary)}", style="bold red")
            for failure in test_result.failures[:_MAX_LISTED_FAILURES]:
                await self._emit(f"  • {_preview(failure, 160)}", style="red")
            if len(test_result.failures) > _MAX_LISTED_FAILURES:
                await self._emit(
                    f"  … and {len(test_result.failures) - _MAX_LISTED_FAILURES} more failure(s)",
                    style="dim",
                )
            await self._emit("Planning fixes for the failures next cycle.", style="yellow")
        return report

    async def _emit_plan(self, plan: Plan, tasks: list[TaskModel]) -> None:
        """Show a compact plan panel: rationale plus one line per task."""
        body = "\n".join(f"[bold]{task.id}[/bold] {task.description}" for task in tasks)
        await self._emit(
            _preview(plan.rationale, 200) if plan.rationale else f"{len(tasks)} task(s) planned",
            renderable=Panel(
                body or "(no tasks -- planner reports the goal complete)",
                title=f"Plan - cycle {self._cycles}",
                border_style="blue",
            ),
        )

    def _planner_input(self) -> str:
        """The original goal plus the latest test failures.

        Failures are appended as context rather than replacing ``self._goal``, so
        the user's goal survives every cycle and lands in ``RunResult.goal``.
        """
        if not self._failures:
            return self._goal
        listed = "\n".join(f"- {_preview(f, 200)}" for f in self._failures[:_MAX_LISTED_FAILURES])
        return (
            f"{self._goal}\n\n"
            f"Context: the previous cycle left {len(self._failures)} failing test(s).\n"
            f"{listed}\n"
            "Plan tasks that fix them, or return an empty task list if the goal is complete."
        )

    def _merge_plan(self, plan: Plan) -> list[TaskModel]:
        """Fold a plan into ``self._tasks`` and return the canonical task objects.

        Tasks are de-duplicated by ``id``. The newest status/result wins, except
        that a task that already settled is never dragged back to pending by a
        later plan.
        """
        canonical: list[TaskModel] = []
        for task in plan.tasks:
            position = self._index.get(task.id)
            if position is None:
                self._index[task.id] = len(self._tasks)
                self._tasks.append(task)
                canonical.append(task)
                continue
            current = self._tasks[position]
            canonical.append(current)
            if current.status in SETTLED_STATUSES and task.status not in SETTLED_STATUSES:
                continue
            current.status = task.status
            if task.result is not None:
                current.result = task.result
            if task.files:
                current.files = task.files
        return canonical

    # ----------------------------------------------------------- execution

    async def _execute(self, tasks: list[TaskModel], report: CycleReport) -> None:
        """Run workers for ``tasks``, bounded by ``settings.max_parallel_tasks``."""
        semaphore = asyncio.Semaphore(max(1, settings.max_parallel_tasks))
        attempts = max(1, settings.max_task_retries + 1)
        results: dict[str, str | BaseException | None] = {}

        async def run_one(task: TaskModel) -> None:
            async with semaphore:
                if self._stop_event.is_set():
                    results[task.id] = None
                    return
                task.status = TaskStatus.RUNNING
                worker = WorkerAgent(task=task, client=self.client)
                for attempt in range(1, attempts + 1):
                    try:
                        result = await worker.run()
                    except Exception as exc:  # noqa: BLE001 - worker failure is data
                        if attempt >= attempts:
                            results[task.id] = exc
                            break
                        results[task.id] = None
                        await self._emit(
                            f"Task {task.id} errored ({_preview(str(exc), 60)}); "
                            f"retrying ({attempt}/{attempts - 1})",
                            style="yellow",
                        )
                        continue
                    results[task.id] = result
                    break

        if tasks:
            await self._race_stop(_gather_all([run_one(task) for task in tasks]))

        for task in tasks:
            outcome = results.get(task.id)
            if isinstance(outcome, BaseException):
                task.status = TaskStatus.FAILED
                task.result = str(outcome)
                report.failed += 1
                await self._emit(
                    f"✗ {task.id} {task.description}: {_preview(str(outcome))}", style="bold red"
                )
            elif outcome is None:
                task.status = TaskStatus.SKIPPED
                task.result = "not run (stopped)" if self._stop_event.is_set() else task.result
                await self._emit(f"· {task.id} {task.description} skipped", style="dim")
            else:
                task.status = TaskStatus.COMPLETED
                task.result = outcome
                report.completed += 1
                await self._emit(
                    f"✓ {task.id} {task.description}: {_preview(outcome)}", style="green"
                )

    async def _race_stop(self, awaitable: Coroutine[Any, Any, _T]) -> _T | None:
        """Await ``awaitable``, returning ``None`` as soon as ``stop()`` is requested."""
        future: asyncio.Future[_T] = asyncio.ensure_future(awaitable)
        stopper: asyncio.Future[bool] = asyncio.ensure_future(self._stop_event.wait())
        # Heterogeneous futures, so pin the element type: bare ``{future, stopper}``
        # infers as ``set[object]``, which ``asyncio.wait`` rejects.
        waitables: set[asyncio.Future[Any]] = {future, stopper}
        try:
            done, _pending = await asyncio.wait(
                waitables, return_when=asyncio.FIRST_COMPLETED
            )
            if future in done:
                return future.result()
            future.cancel()
            try:
                await future
            except asyncio.CancelledError:
                # Expected on the stop path: we cancelled it ourselves.
                logger.debug("stop(): in-flight work cancelled cleanly")
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.warning(
                    "stop(): in-flight work raised during cleanup: %s: %s",
                    type(exc).__name__,
                    exc,
                )
            return None
        finally:
            stopper.cancel()
            if not future.done():
                future.cancel()


async def _gather_all(coros: list[Coroutine[Any, Any, None]]) -> None:
    """Run every coroutine, swallowing their exceptions (collected by the caller)."""
    await asyncio.gather(*coros, return_exceptions=True)


def _preview(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    """Single-line, length-capped excerpt of a worker result or test summary."""
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    return flat[: max(1, limit - 3)] + "..."
