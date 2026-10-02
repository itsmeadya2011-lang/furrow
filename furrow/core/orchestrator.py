"""The run loop of furrow: plan, execute in dependency waves, test, repeat.

:class:`Orchestrator` owns everything a run needs to judge progress: the cumulative
task list, the plan of the current cycle, the test results, the emitted events and the
reason the loop finally stopped.  A run is a sequence of cycles; each cycle plans work
for the *original* goal, executes the planned tasks concurrently in dependency order,
and runs the test suite.  Failures never end the run by themselves, but the loop is
always bounded: a cycle that completes no task counts as stalled, two consecutive
stalled cycles stop it, and a task is abandoned after ``settings.max_task_attempts``
failed attempts.  Every run ends with a :class:`~furrow.config.RunResult`.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterator
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.status import Status
from rich.table import Table

from furrow import config as config_module
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, RunResult, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.events import Event, EventCallback, EventKind, emit, make_event
from furrow.llm import LLMClient
from furrow.util import truncate

__all__ = ["Orchestrator"]

console = Console()

_MAX_EVENTS = 1000
_MAX_CONSECUTIVE_ERRORS = 3
_MAX_STALLED_CYCLES = 2
_CELL_LIMIT = 200
_FOCUS_LIMIT = 1000


def _cell(value: object, limit: int = _CELL_LIMIT) -> str:
    """Render ``value`` for rich: truncated, and with markup characters neutralized.

    Everything interpolated into console output goes through here; an error message or
    file name containing ``[bold]`` would otherwise be interpreted as markup and abort
    the run with a :class:`rich.errors.MarkupError`.
    """
    return escape(truncate(str(value), limit))


def _status_value(status: TaskStatus | str) -> str:
    """Return the plain string value of a task status."""
    return status.value if isinstance(status, TaskStatus) else str(status)


def _task_done(task: TaskModel) -> bool:
    """Return whether ``task`` needs no more work (done now, or done in an earlier cycle)."""
    return task.status in (TaskStatus.COMPLETED, TaskStatus.SKIPPED)


def _waves(tasks: list[TaskModel]) -> tuple[list[list[TaskModel]], list[str]]:
    """Group ``tasks`` into waves of mutually independent work.

    A task belongs to the first wave whose dependencies are all satisfied by an earlier
    wave, so the waves are a topological order of the planned work and each wave can be
    executed concurrently.

    Args:
        tasks: The tasks of one plan, in planner order.

    Returns:
        The waves in execution order, plus human readable notes about dependencies that
        could not be resolved (unknown ids, self references, cycles).  Anything
        unresolvable is placed in one final wave so a malformed plan still runs instead
        of hanging or raising.
    """
    notes: list[str] = []
    known = {task.id for task in tasks}
    blocked_by: dict[str, list[str]] = {}
    for task in tasks:
        deps: list[str] = []
        for dep in task.dependencies:
            if dep == task.id:
                notes.append(f"task {task.id} depends on itself; ignoring that dependency")
            elif dep not in known:
                notes.append(f"task {task.id} depends on unknown id {dep!r}; ignoring it")
            else:
                deps.append(dep)
        blocked_by[task.id] = deps

    satisfied: set[str] = set()
    remaining = list(tasks)
    waves: list[list[TaskModel]] = []
    while remaining:
        wave: list[TaskModel] = []
        still_blocked: list[TaskModel] = []
        for task in remaining:
            target = wave if all(dep in satisfied for dep in blocked_by[task.id]) else still_blocked
            target.append(task)
        if not wave:
            break
        waves.append(wave)
        satisfied.update(task.id for task in wave)
        remaining = still_blocked

    if remaining:
        ids = ", ".join(task.id for task in remaining)
        notes.append(f"cyclic or unsatisfiable dependencies ({ids}); running them last")
        waves.append(remaining)
    return waves, notes


class Orchestrator:
    """Runs the plan/execute/test loop until the goal is reached or the run stops.

    Args:
        goal: The objective of the run.  It is never rewritten; later cycles refine the
            work, not the goal.
        client: LLM client to use.  One is created from ``settings`` when omitted.
        settings: Runtime configuration; defaults to the client settings (or the
            process-wide settings when a client is created).
        on_event: Optional sync/async callback receiving every :class:`Event`.

    Attributes:
        goal: The original goal, verbatim.
        focus: Failure feedback handed to the next planning round, or ``None``.
        cycles: Number of cycles started so far.
        plan: The plan of the most recent cycle, or ``None``.
        tasks: Every task the run has planned, cumulative across cycles, each appearing
            once (matched on description) with its latest status.
        test_results: One :class:`TestResult` per executed test run.
        events: The last 1000 events emitted, in order.
        completed: Whether the goal was reached.
        stop_reason: Why the run stopped.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        self.on_event = on_event
        self.client: LLMClient = client if client is not None else LLMClient(settings=settings)
        self.settings: Settings = (
            settings
            if settings is not None
            else getattr(self.client, "settings", None) or config_module.settings
        )

        self.focus: str | None = None
        self.cycles: int = 0
        self.plan: Plan | None = None
        self.tasks: list[TaskModel] = []
        self.test_results: list[TestResult] = []
        self.events: list[Event] = []
        self.completed: bool = False
        self.stop_reason: str = ""

        self._stopping = False
        self._consecutive_errors = 0
        self._last_error = ""
        self._stalled_cycles = 0
        self._cycle_completed = 0
        self._last_test: TestResult | None = None

    async def run(self) -> RunResult:
        """Run cycles until the goal is complete, a stop condition fires, or ``stop()``.

        Stop conditions are checked between cycles and between task waves: a cycle limit
        (``settings.max_cycles``), three consecutive cycles that raised, two consecutive
        cycles that completed no task (``no progress``), a planner that reports no
        remaining work, and a goal reached with every task completed and the tests passing
        (or not runnable at all).  A cycle is judged *stalled* by its own work, not by the
        test verdict: green tests over tasks that all failed still count as no progress,
        which is what keeps ``max_cycles=0`` from looping forever.  The terminal reasons are
        evaluated before the stall guard, so a cycle that converges by only skipping work an
        earlier cycle already finished is still reported as complete.

        Returns:
            A :class:`RunResult` describing the run.  Ordinary agent failures are
            reported through events and the result, never raised; ``KeyboardInterrupt``
            and ``asyncio.CancelledError`` propagate.
        """
        self._print_header()
        while not self._stopping:
            limit = self.settings.max_cycles
            if limit > 0 and self.cycles >= limit:
                self.stop_reason = "cycle limit reached"
                break

            self.cycles += 1
            await self._emit(EventKind.CYCLE, f"cycle {self.cycles} started")
            try:
                await self._cycle()
            except Exception as exc:  # PlannerError, LLMError, ... : reported, not fatal
                await self._record_cycle_error(exc)
                if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                    break
                continue
            self._consecutive_errors = 0
            if self._stopping:
                break  # state is partial; do not judge completion on it

            # Terminal reasons first: a cycle that finished the goal off is a success even
            # when it completed no task itself.
            if self.plan is not None and not self.plan.tasks:
                if self._last_test is not None and self._last_test.passed:
                    self.completed = True
                    self.stop_reason = "planner reported no remaining work"
                    break
            elif self._goal_complete():
                self.completed = True
                self.stop_reason = "goal complete"
                break

            if self._cycle_completed > 0:
                self._stalled_cycles = 0
            else:
                self._stalled_cycles += 1
                if self._stalled_cycles >= _MAX_STALLED_CYCLES:
                    self.stop_reason = "no progress"
                    break

        return await self._finish()

    async def stop(self) -> None:
        """Ask the run to stop at the next checkpoint.

        Cooperative and safe to call from another task or a signal handler: it only sets
        a flag, which the loop checks between cycles and between task waves.
        """
        self._stopping = True
        if not self.stop_reason:
            self.stop_reason = "stop requested"

    async def _cycle(self) -> None:
        """Run one plan/execute/test round.

        Raises:
            Exception: Anything the planner or tester raises; the caller counts it as a
                failed cycle.
        """
        self._cycle_completed = 0
        if self._stopping:
            return

        plan = await self._plan()
        self.plan = plan
        self._print_plan(plan)
        tasks = self._merge(plan.tasks)
        # ``self.plan`` holds the objects that actually ran, so their live statuses are
        # what the completion check and the test report see.
        self.plan = plan.model_copy(update={"tasks": tasks})
        if tasks:
            await self._execute(tasks)
            self._print_tasks(tasks)
            if self._stopping:
                return
        await self._test(tasks)

    async def _plan(self) -> Plan:
        """Ask the planner for the next batch of work on the original goal."""
        planner = PlannerAgent(client=self.client, settings=self.settings)
        with self._spinner("[bold yellow]Planning...[/bold yellow]"):
            plan = await planner.plan(self.goal, focus=self.focus, context=self._context())
        await self._emit(
            EventKind.PLAN,
            f"{len(plan.tasks)} task(s) planned",
            data={"count": len(plan.tasks), "tasks": [task.id for task in plan.tasks]},
        )
        return plan

    def _context(self) -> str:
        """Build the short state blob handed to the planner alongside the goal."""
        lines: list[str] = []
        if self.focus:
            lines.append("Previous cycle findings:")
            lines.append(truncate(self.focus, _FOCUS_LIMIT))
        done = [task for task in self.tasks if task.status == TaskStatus.COMPLETED]
        if done:
            lines.append("Already completed:")
            lines.extend(
                f"- {truncate(task.description, _CELL_LIMIT)}"
                for task in done[-max(1, self.settings.planner_max_tasks) :]
            )
        if self._last_test is not None:
            verdict = "passed" if self._last_test.passed else "failed"
            lines.append(f"Last test run {verdict}: {truncate(self._last_test.summary, _CELL_LIMIT)}")
        return "\n".join(lines)

    def _merge(self, planned: list[TaskModel]) -> list[TaskModel]:
        """Fold newly planned tasks into the cumulative :attr:`tasks` list.

        Tasks are matched on description so work already done in an earlier cycle is not
        re-implemented: the previous task object is reused (keeping its status) and
        therefore reported as skipped.  A previously failed task is reset so the cycle
        can retry it -- :meth:`_run_task` is what stops a task that used up
        ``settings.max_task_attempts`` from being dispatched again.

        Returns:
            The task objects to execute, one per planned task.
        """
        known = {task.description: task for task in self.tasks}
        claimed: set[int] = set()
        merged: list[TaskModel] = []
        for task in planned:
            prior = known.get(task.description)
            if prior is None:
                known[task.description] = task
                self.tasks.append(task)
            else:
                prior.files = task.files or prior.files
                prior.dependencies = task.dependencies
                if prior.status == TaskStatus.FAILED:
                    prior.status = TaskStatus.PENDING
                    prior.error = None
                if id(prior) in claimed:
                    task = prior.model_copy(
                        deep=True,
                        update={
                            "status": TaskStatus.PENDING,
                            "result": None,
                            "error": None,
                            "attempts": 0,
                        },
                    )
                else:
                    task = prior
            claimed.add(id(task))
            merged.append(task)
        return merged

    async def _execute(self, tasks: list[TaskModel]) -> None:
        """Execute ``tasks`` wave by wave, concurrently inside each wave.

        Tasks whose description was already completed in an earlier cycle are skipped.
        Concurrency is bounded by ``settings.max_parallel_tasks``; a non-positive setting
        falls back to one task at a time.
        """
        waves, notes = _waves(tasks)
        for note in notes:
            await self._emit(EventKind.LOG, note)
            console.print(f"[yellow]{_cell(note, _FOCUS_LIMIT)}[/yellow]")

        done_descriptions = {
            task.description for task in self.tasks if task.status == TaskStatus.COMPLETED
        }
        limit = self.settings.max_parallel_tasks
        if limit <= 0:
            limit = 1
        semaphore = asyncio.Semaphore(limit)

        for wave in waves:
            if self._stopping:
                await self._emit(
                    EventKind.LOG, f"stop requested: {len(wave)} task(s) not started"
                )
                console.print(f"[yellow]{_cell('stop requested: skipping remaining tasks')}[/yellow]")
                return
            with self._spinner(f"[bold yellow]Running {len(wave)} task(s)...[/bold yellow]"):
                results = await asyncio.gather(
                    *(self._run_task(task, done_descriptions, semaphore) for task in wave),
                    return_exceptions=True,
                )
            for task, result in zip(wave, results, strict=True):
                if isinstance(result, BaseException) and not isinstance(result, Exception):
                    raise result
                if isinstance(result, Exception):  # defensive: _run_task records its own
                    await self._fail_task(task, result)

    async def _run_task(
        self, task: TaskModel, done_descriptions: set[str], semaphore: asyncio.Semaphore
    ) -> str | None:
        """Run one task, recording status, attempts, result or error.

        A task that has already used up ``settings.max_task_attempts`` dispatches is
        abandoned instead of retried: it is marked ``FAILED`` and never handed to a worker
        again.  That, plus the stall guard in :meth:`run`, is what bounds the loop when the
        planner keeps re-planning work that can never succeed.

        Returns:
            The task id when the task completed, ``None`` when it was skipped, failed or
            abandoned.
        """
        if task.description in done_descriptions:
            task.status = TaskStatus.SKIPPED
            await self._emit(
                EventKind.TASK_SKIPPED,
                "already completed in an earlier cycle",
                task_id=task.id,
                data={"summary": truncate(task.description, _CELL_LIMIT)},
            )
            return None
        if task.attempts >= self.settings.max_task_attempts:
            await self._abandon_task(task)
            return None

        task.status = TaskStatus.RUNNING
        task.attempts += 1
        await self._emit(
            EventKind.TASK_START, truncate(task.description, _CELL_LIMIT), task_id=task.id
        )
        try:
            async with semaphore:
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                output = await worker.run(on_event=self.on_event)
        except Exception as exc:
            await self._fail_task(task, exc)
            return None

        task.status = TaskStatus.COMPLETED
        task.result = truncate(output or "", self.settings.max_output_chars)
        task.error = None
        self._cycle_completed += 1
        await self._emit(
            EventKind.TASK_DONE,
            truncate(task.description, _CELL_LIMIT),
            task_id=task.id,
            data={"files": len(task.files), "summary": truncate(output or "", _CELL_LIMIT)},
        )
        return task.id

    async def _fail_task(self, task: TaskModel, exc: BaseException) -> None:
        """Mark ``task`` failed and announce it as ``TASK_FAILED``."""
        message = truncate(f"{type(exc).__name__}: {exc}", self.settings.max_output_chars)
        task.status = TaskStatus.FAILED
        task.error = message
        await self._emit(
            EventKind.TASK_FAILED,
            truncate(message, _CELL_LIMIT),
            task_id=task.id,
            data={"summary": truncate(message, _CELL_LIMIT)},
        )
        console.print(f"[red]Task {_cell(task.id)} failed: {_cell(message, _CELL_LIMIT)}[/red]")

    async def _abandon_task(self, task: TaskModel) -> None:
        """Give up on ``task``: it is ``FAILED`` and will not be dispatched again.

        Announced as ``TASK_FAILED`` like any other failure -- the task is not done, so it
        counts towards ``RunResult.tasks_failed`` -- with the last error kept as context.
        """
        last_error = task.error or "no error recorded"
        message = (
            f"abandoned after {task.attempts} attempts: "
            f"{truncate(last_error, _CELL_LIMIT)}"
        )
        task.status = TaskStatus.FAILED
        task.error = message
        await self._emit(
            EventKind.TASK_FAILED,
            truncate(message, _CELL_LIMIT),
            task_id=task.id,
            data={
                "summary": truncate(message, _CELL_LIMIT),
                "attempts": task.attempts,
                "abandoned": True,
            },
        )
        console.print(f"[red]Task {_cell(task.id)} {_cell(message, _CELL_LIMIT)}[/red]")

    async def _test(self, tasks: list[TaskModel]) -> None:
        """Run the test suite against the work of this cycle and record the verdict."""
        tester = TesterAgent(client=self.client, settings=self.settings)
        with self._spinner("[bold yellow]Testing...[/bold yellow]"):
            result = await tester.run(self.goal, tasks)

        self.test_results.append(result)
        self._last_test = result
        summary = truncate(result.summary, _FOCUS_LIMIT)
        await self._emit(
            EventKind.TESTS,
            summary,
            data={
                "passed": result.passed,
                "summary": summary,
                "command": result.command,
                "failures": len(result.failures),
            },
        )
        if not result.passed:
            self.focus = truncate("\n".join(result.failures) or result.summary, _FOCUS_LIMIT)
        self._print_tests(result)

    def _goal_complete(self) -> bool:
        """Return whether the current plan is fully done and the tests are green.

        A test run that never produced a command counts as unverifiable rather than
        failing, so a project without a test runner can still complete.
        """
        plan = self.plan
        if plan is None or not plan.tasks or not all(_task_done(t) for t in plan.tasks):
            return False
        test = self._last_test
        if test is None:
            return False
        return test.passed or test.command is None

    async def _record_cycle_error(self, exc: BaseException) -> None:
        """Report a cycle that raised as an ``ERROR`` event and count it."""
        self._consecutive_errors += 1
        self._last_error = truncate(f"{type(exc).__name__}: {exc}", _FOCUS_LIMIT)
        if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS and not self.stop_reason:
            self.stop_reason = f"repeated cycle errors: {self._last_error}"
        await self._emit(
            EventKind.ERROR,
            self._last_error,
            data={"summary": self._last_error, "count": self._consecutive_errors},
        )
        console.print(f"[red]Cycle failed: {_cell(self._last_error, _FOCUS_LIMIT)}[/red]")

    async def _finish(self) -> RunResult:
        """Build, render and announce the :class:`RunResult` of this run."""
        result = RunResult(
            goal=self.goal,
            cycles=self.cycles,
            completed=self.completed,
            reason=self.stop_reason or "stopped",
            tests_passed=bool(self._last_test is not None and self._last_test.passed),
            # Every task that ended the run in a terminal done state counts, so the number
            # agrees with ``completed`` even when the last cycle only skipped work an
            # earlier cycle had already finished.
            tasks_completed=sum(1 for task in self.tasks if _task_done(task)),
            tasks_failed=sum(1 for task in self.tasks if task.status == TaskStatus.FAILED),
        )
        if self.on_event is None:
            # No consumer to render for, so the run prints its own summary panel.  With an
            # ``on_event`` callback the consumer owns rendering (the CLI prints its own
            # panel when it sees ``DONE``) and printing here too would duplicate it.
            self._print_summary(result)
        await self._emit(
            EventKind.DONE,
            result.reason,
            data={
                "completed": result.completed,
                "reason": result.reason,
                "cycles": result.cycles,
                "tasks_completed": result.tasks_completed,
                "tasks_failed": result.tasks_failed,
                "tests_passed": result.tests_passed,
            },
        )
        return result

    async def _emit(
        self,
        kind: EventKind | str,
        message: str = "",
        *,
        task_id: str | None = None,
        data: dict[str, Any] | None = None,
    ) -> Event:
        """Build an event stamped with the current cycle, record it and hand it out.

        The recorded history is bounded to the last 1000 events.
        """
        event = make_event(EventKind(kind), message, cycle=self.cycles, task_id=task_id, data=data)
        self.events.append(event)
        if len(self.events) > _MAX_EVENTS:
            del self.events[: len(self.events) - _MAX_EVENTS]
        await emit(self.on_event, event)
        return event

    @contextlib.contextmanager
    def _spinner(self, text: str) -> Iterator[None]:
        """Show a status line while a long await runs; a no-op when not on a terminal."""
        if console.is_terminal:
            with Status(text, console=console):
                yield
        else:
            yield

    def _print_header(self) -> None:
        body = f"[bold green]Furrow[/bold green]\nGoal: {_cell(self.goal, _FOCUS_LIMIT)}"
        console.print(Panel.fit(body, title="Furrow", border_style="green"))

    def _print_plan(self, plan: Plan) -> None:
        table = Table(title=f"Cycle {self.cycles} plan", border_style="blue")
        for column in ("id", "description", "files", "needs"):
            table.add_column(column)
        for task in plan.tasks:
            table.add_row(
                _cell(task.id),
                _cell(task.description),
                _cell(len(task.files)),
                _cell(", ".join(task.dependencies)) if task.dependencies else "-",
            )
        if plan.rationale:
            table.caption = _cell(plan.rationale, _FOCUS_LIMIT)
        console.print(table)

    def _print_tasks(self, tasks: list[TaskModel]) -> None:
        table = Table(title=f"Cycle {self.cycles} tasks", border_style="cyan")
        for column in ("id", "status", "description", "detail"):
            table.add_column(column)
        for task in tasks:
            table.add_row(
                _cell(task.id),
                _cell(_status_value(task.status)),
                _cell(task.description),
                _cell(task.error or task.result or ""),
            )
        console.print(table)

    def _print_tests(self, result: TestResult) -> None:
        style, label = ("green", "Tests passed") if result.passed else ("red", "Tests failed")
        command = _cell(result.command) if result.command else "n/a"
        console.print(
            f"[{style}]{label}[/{style}] [dim]({command})[/dim]: "
            f"{_cell(result.summary, _FOCUS_LIMIT)}"
        )
        for failure in result.failures[:10]:
            console.print(f"  [red]•[/red] {_cell(failure, _FOCUS_LIMIT)}")

    def _print_summary(self, result: RunResult) -> None:
        lines = [
            f"Goal: {_cell(result.goal, _FOCUS_LIMIT)}",
            f"Cycles: {_cell(result.cycles)}",
            f"Tasks completed: {_cell(result.tasks_completed)}",
            f"Tasks failed: {_cell(result.tasks_failed)}",
            f"Tests: {'passed' if result.tests_passed else 'not passing'}",
            f"Reason: {_cell(result.reason)}",
        ]
        border = "green" if result.completed else "yellow"
        console.print(Panel("\n".join(lines), title="Run summary", border_style=border))
