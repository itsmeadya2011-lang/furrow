"""Core orchestration loop for Furrow.

Furrow is an autonomous coding agent: it repeatedly asks a planner for a plan,
executes that plan with a pool of workers, runs the test suite, and folds the
outcome back into the next plan.  The loop is "infinite" in the sense that it
is not limited to a fixed number of steps, but it is bounded and interruptible.

Termination conditions (all enforced by :meth:`Orchestrator.run`):

1. **Goal met** - every task of the current plan completed *and* the tester
   reported success, or the planner proposed no further work while the tests
   were already green.  ``run()`` returns ``True``.
2. **Stop requested** - :meth:`Orchestrator.stop` raises a flag that is honoured
   at every phase boundary and before every worker start, so ``run()`` returns
   ``False`` without ever starting a new cycle.
3. **Cycle budget exhausted** - ``settings.max_cycles > 0`` bounds the loop.
   ``0`` means unlimited.
4. **Unrecoverable agent error** - a planner (or tester) failure is reported
   through the console and the event callback and halts the loop cleanly instead
   of raising a traceback at the caller.
5. **Cancellation** - ``asyncio.CancelledError`` is cleaned up and re-raised, as
   asyncio requires.

State kept across cycles: the immutable original goal (:attr:`Orchestrator.goal`),
the planner focus derived for the next cycle (:attr:`Orchestrator.current_focus`),
every plan produced so far (:attr:`Orchestrator.history`), the current plan whose
task ``status`` fields are mutated in place (:attr:`Orchestrator.current_plan`)
and the latest verdict of the test suite
(:attr:`Orchestrator.last_test_result`).

One run per instance
--------------------

:meth:`Orchestrator.run` is single-use: an instance carries the state of exactly
one run, and ``run()`` neither clears nor re-arms it.  Construct a new
:class:`Orchestrator` to run again.

Every run ends with exactly one ``done`` event, whether the goal was met or the
loop halted for one of the reasons above, and *message* is the reason.  A
transport that adds frames of its own must not reuse that name for its own
terminal frame, or a client keyed on ``done`` cannot tell the two apart.

Task statuses
-------------

A task starts out ``PENDING`` and the loop only ever writes ``COMPLETED``,
``FAILED`` or ``SKIPPED``.  ``SKIPPED`` covers two outcomes that one status
cannot separate, so the reason is carried by ``TaskModel.result`` under one of
two stable prefixes (:data:`SKIPPED_CANCELLED_PREFIX`,
:data:`SKIPPED_BLOCKED_PREFIX`):

* ``not run: <reason>`` - the task was **cancelled**: a stop was requested
  before it started, either for a whole remaining wave
  (:meth:`Orchestrator._skip_pending`) or for a single worker queued behind a
  stop (:meth:`Orchestrator._report_task`);
* ``skipped: dependency <id> (<status>) did not complete`` - the task was
  **blocked**: a dependency failed, was itself skipped or never ran, so
  running it would have been wrong.

A consumer that watches ``status`` alone therefore cannot tell a cancelled run
from a blocked dependency, and must read the ``result`` prefix to do so.  See
:data:`TaskStatus` in :mod:`furrow.config` for why the enum has no separate
member for either case.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator
from enum import Enum
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TaskStatus, TestResult
from furrow.config import settings as global_settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.config import Settings

EventCallback = Callable[[str, str], None]
"""``(event_name, human_readable_message)`` sink, e.g. a websocket broadcaster."""

console = Console()

# Task status vocabulary.  ``TaskModel`` starts out as PENDING; the loop only
# ever writes COMPLETED, FAILED or SKIPPED.
PENDING = TaskStatus.PENDING
COMPLETED = TaskStatus.COMPLETED
FAILED = TaskStatus.FAILED
SKIPPED = TaskStatus.SKIPPED

# ``SKIPPED`` means two different things, so the free-text result starts with one
# of these two prefixes to say which.  They are part of the contract a consumer
# reads ``task.result`` for, not incidental wording; see the module docstring.
SKIPPED_CANCELLED_PREFIX = "not run: "
"""``TaskModel.result`` prefix: the run was stopped before this task started."""

SKIPPED_BLOCKED_PREFIX = "skipped: "
"""``TaskModel.result`` prefix: a dependency did not complete, so this was blocked."""

# Marker for a worker that was never started because a stop was requested.  A
# distinct object (rather than ``None``) keeps "not run" distinguishable from a
# worker that legitimately produced no text.
_NOT_RUN = object()

# Keeps the planner prompt bounded when the loop runs for a very long time.
_MAX_HISTORY_ITEMS = 20
_MAX_FAILURE_ITEMS = 10
_MAX_TEXT = 160


class CycleOutcome(str, Enum):
    """How a single plan -> execute -> test cycle ended."""

    COMPLETED = "completed"
    """All tasks completed and the tester passed."""
    NOTHING_PLANNED = "nothing_planned"
    """The planner proposed no work and the tests were already green."""
    TESTS_FAILED = "tests_failed"
    """Something is still outstanding; another cycle may help."""
    STOPPED = "stopped"
    """``stop()`` was called while the cycle was in flight."""
    PLANNER_ERROR = "planner_error"
    """The planner raised; the run halts cleanly."""


def _task_id(task: TaskModel) -> str:
    return str(getattr(task, "id", "") or "")


def _status(task: TaskModel) -> str:
    return str(getattr(task, "status", PENDING) or PENDING)


def _dependencies(task: TaskModel) -> list[str]:
    return [str(dep) for dep in (getattr(task, "dependencies", None) or []) if str(dep)]


def _one_line(text: object, limit: int = _MAX_TEXT) -> str:
    """Collapse ``text`` to a single truncated line, for events and history."""
    flat = " ".join(str(text or "").split())
    if len(flat) > limit:
        return flat[: max(1, limit - 3)] + "..."
    return flat


def _dependency_waves(tasks: list[TaskModel]) -> tuple[list[list[TaskModel]], list[str]]:
    """Group ``tasks`` into waves that may be executed in parallel.

    A task becomes runnable once all of its dependencies sit in an earlier wave.
    The planner is an LLM, so the graph is handled defensively:

    * a dependency naming an unknown task id is ignored, i.e. treated as
      satisfied, so a hallucinated id cannot stall the whole plan;
    * a task depending on itself has that self-reference dropped;
    * tasks sharing an id are released together, so a dependency on that id
      resolves as one unit;
    * a genuine cycle is broken rather than deadlocked: the remaining tasks are
      released as one final wave and their ids are returned alongside so the
      caller can report the anomaly.
    """
    groups: dict[str, list[TaskModel]] = {}
    for task in tasks:
        groups.setdefault(_task_id(task), []).append(task)

    dependencies: dict[str, list[str]] = {}
    for task_id, members in groups.items():
        collected: list[str] = []
        for member in members:
            for dep in _dependencies(member):
                if dep not in groups or dep == task_id or dep in collected:
                    continue
                collected.append(dep)
        dependencies[task_id] = collected

    pending = list(groups)
    waves: list[list[TaskModel]] = []
    forced: list[str] = []
    while pending:
        remaining = set(pending)
        ready = [t for t in pending if not any(d in remaining for d in dependencies[t])]
        if not ready:
            # Every remaining task waits on another one: release them all.
            ready = list(pending)
            forced = list(pending)
        ready_set = set(ready)
        waves.append([task for task_id in ready for task in groups[task_id]])
        pending = [task_id for task_id in pending if task_id not in ready_set]
    return waves, forced


class Orchestrator:
    """Drives plan/execute/test cycles until a goal is met or the loop halts.

    The LLM client, the settings and the event callback are injected and no
    network call happens in :meth:`__init__`, so a run can be driven in tests by
    swapping the agent classes for stubs.

    An instance drives exactly one :meth:`run`; see the module docstring for what
    that means for :meth:`run`, :meth:`stop` and the status vocabulary.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        on_event: EventCallback | None = None,
    ) -> None:
        self.goal = goal
        """The original goal. Never rewritten; see :attr:`current_focus`."""
        self.settings = settings or global_settings
        self.on_event = on_event
        self.client = client or LLMClient(settings=self.settings)
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.current_focus: str = goal
        """The prompt handed to the planner for the next cycle."""
        self.history: list[Plan] = []
        """Plans of previous cycles, in order; task statuses are mutated in place."""
        self.last_test_result: TestResult | None = None
        """Verdict of the most recent test run, or ``None`` before the first one."""
        self.results: dict[str, str] = {}
        """Task id -> result text for the current cycle."""
        self.error: str | None = None
        """The planner/tester error that halted the run, if any."""
        self._current_plan: Plan | None = None
        self._cycles = 0
        self._done = False
        self._stop = asyncio.Event()
        self._consumer_warned = False

    # ------------------------------------------------------------------ public

    async def run(self) -> bool:
        """Run cycles until the goal is met, a stop is requested, or the budget ends.

        Returns ``True`` only when the goal was actually met.  A graceful stop,
        an exhausted ``max_cycles`` budget and an agent error all return
        ``False``.  See the module docstring for every termination condition.

        **One ``run()`` per instance.**  This method is single-use: it does not
        reset :attr:`_stop`, :attr:`_cycles` or :attr:`history`, so a second call
        is not a second run.  It returns ``False`` immediately after a
        cooperative :meth:`stop`, and halts on the very first budget check after
        a run that used up ``max_cycles`` (``_cycles`` is still at the budget).

        The state is deliberately cumulative rather than cleared: after the call
        :attr:`history` and :attr:`cycles` describe *the* run, which is the whole
        point of collecting them, and :attr:`_stop` is sticky so a stop asked for
        by a caller - or by a worker - can never be forgotten and quietly
        re-armed by a second call.  Resetting instead would trade an accurate
        record for a silent, surprising second run.

        To run again, build a new :class:`Orchestrator` for the new goal.  Both
        in-repo callers already do: the CLI constructs one per invocation and
        the web layer one per WebSocket connection.
        """
        console.print(
            Panel.fit(
                f"[bold green]Furrow[/bold green]\nGoal: {self.goal}",
                title="Furrow",
            )
        )
        self._emit("run", f"Starting Furrow on goal: {self.goal}")
        try:
            while not self._stop.is_set():
                budget = self.settings.max_cycles
                if budget and self._cycles >= budget:
                    self._halt(f"Reached max_cycles limit ({budget}). Halting.")
                    return False
                self._cycles += 1
                console.print(f"\n[bold cyan]═══ Cycle {self._cycles} ═══[/bold cyan]")
                if budget:
                    self._emit("cycle", f"Cycle {self._cycles} of {budget} started.")
                else:
                    self._emit("cycle", f"Cycle {self._cycles} started.")
                outcome = await self._cycle()
                if self._stop.is_set():
                    self._halt("Stop requested. Halting.")
                    return False
                if outcome in (CycleOutcome.COMPLETED, CycleOutcome.NOTHING_PLANNED):
                    self._done = True
                    console.print("[bold green]Goal complete. Halting.[/bold green]")
                    self._emit("done", "Goal complete.")
                    return True
                if outcome is CycleOutcome.PLANNER_ERROR:
                    self._halt(f"Planner failed: {self.error}. Halting.")
                    return False
                # TESTS_FAILED: re-plan with a focus built from this cycle.
            self._halt("Stop requested. Halting.")
            return False
        except asyncio.CancelledError:
            self._stop.set()
            self._done = False
            console.print("[yellow]Furrow cancelled.[/yellow]")
            self._emit("done", "Run cancelled.")
            raise

    async def stop(self) -> None:
        """Request a graceful stop of a running :meth:`run`.

        The request is cooperative: the flag is checked before every cycle, at
        every phase boundary and before each worker starts, so an in-flight wave
        still finishes (workers are never killed mid-write) but no new work is
        started and no further cycle is planned.  Idempotent, and safe to call
        when no run is active.

        The request is also sticky: it is never cleared, not by the run that
        honours it and not by anything else.  Since :meth:`run` is one-shot per
        instance, a stopped orchestrator is spent - a second ``run()`` would
        return ``False`` at once.  Build a new :class:`Orchestrator` to run
        again, which is what both in-repo callers do.
        """
        self._stop.set()

    @property
    def done(self) -> bool:
        """``True`` once the goal has been met, i.e. ``run()`` returned ``True``."""
        return self._done

    @property
    def cycles(self) -> int:
        """Number of cycles started so far."""
        return self._cycles

    @property
    def current_plan(self) -> Plan | None:
        """The plan of the most recent cycle, with task statuses mutated in place."""
        return self._current_plan

    # ------------------------------------------------------------------ cycle

    async def _cycle(self) -> CycleOutcome:
        """Run one plan -> execute -> test pass and report how it ended."""
        self._done = False
        self.results = {}
        self.error = None
        self.current_focus = self._build_focus()
        head = self.current_focus.splitlines()[0]
        console.print(f"[dim]Focus: {head}[/dim]")
        self._emit("plan", f"Planning cycle {self._cycles}: {head}")

        try:
            with self._spinner("Planning..."):
                raw_plan = await self.planner.plan(self.current_focus)
            plan = self._as_plan(raw_plan)
        except Exception as exc:  # noqa: BLE001 - a broken planner must not kill the run
            self.error = f"{type(exc).__name__}: {exc}"
            console.print(f"[red]Planner error: {self.error}[/red]")
            self._emit("error", f"Planner error: {self.error}")
            return CycleOutcome.PLANNER_ERROR

        self._current_plan = plan
        # Tasks are mutated in place, so history reflects final statuses.
        self.history.append(plan)
        console.print(Panel(self._plan_view(plan), title="Plan", border_style="blue"))
        rationale = _one_line(getattr(plan, "rationale", ""), 300)
        if rationale:
            self._emit("plan", f"Plan for cycle {self._cycles}: {rationale}")

        tasks = list(plan.tasks)
        if tasks:
            await self._execute_plan(tasks)
        else:
            console.print("[yellow]Planner proposed no tasks; verifying with the tester.[/yellow]")
            self._emit("plan", "Planner proposed no tasks this cycle.")
        if self._stop.is_set():
            return CycleOutcome.STOPPED

        self.last_test_result = await self._run_tester(tasks)
        if not self.last_test_result.passed:
            summary = self.last_test_result.summary
            console.print(f"[red]Tests failed: {summary}[/red]")
            for failure in self.last_test_result.failures:
                console.print(f"  • {_one_line(failure, 400)}")
            self._emit("test", f"Tests failed: {summary}")
            console.print("[yellow]Will attempt fix in next cycle.[/yellow]")
            self.current_focus = self._build_focus()
            return CycleOutcome.TESTS_FAILED

        summary = self.last_test_result.summary
        console.print(f"[green]Tests passed: {summary}[/green]")
        self._emit("test", f"Tests passed: {summary}")
        if not tasks:
            # Nothing was left to do and the suite is green: the goal is met.
            return CycleOutcome.NOTHING_PLANNED
        if not self._is_done():
            console.print("[yellow]Plan incomplete; re-planning remaining work.[/yellow]")
            self.current_focus = self._build_focus()
            return CycleOutcome.TESTS_FAILED
        return CycleOutcome.COMPLETED

    async def _execute_plan(self, tasks: list[TaskModel]) -> None:
        """Execute ``tasks`` in dependency waves with bounded parallelism.

        Tasks whose dependencies all completed run together, capped at
        ``settings.max_parallel_tasks`` by a semaphore.  A task whose dependency
        failed, was skipped or never ran is marked ``skipped`` and is not run at
        all: its result starts with :data:`SKIPPED_BLOCKED_PREFIX`, naming the
        dependency and its status.  When a stop is requested the tasks that were
        never started are marked the same way but with
        :data:`SKIPPED_CANCELLED_PREFIX` instead, so the two reasons stay
        distinguishable - see the module docstring.  Workers never print: their
        results are collected and reported from the single place below, so
        concurrent completions cannot interleave with the Rich console or with
        the event stream.
        """
        waves, forced = _dependency_waves(tasks)
        if forced:
            note = (
                f"Dependency cycle between tasks {', '.join(forced)}; "
                "releasing them together to break it."
            )
            console.print(f"[yellow]{note}[/yellow]")
            self._emit("task", note)

        limit = max(1, int(self.settings.max_parallel_tasks))
        semaphore = asyncio.Semaphore(limit)
        for index, wave in enumerate(waves, start=1):
            if self._stop.is_set():
                self._skip_pending(wave, "stop requested")
                return
            runnable, blocked = self._partition(wave, tasks, set(forced))
            for task, reason in blocked:
                self._mark(
                    task,
                    SKIPPED,
                    f"{SKIPPED_BLOCKED_PREFIX}dependency {reason} did not complete",
                )
                console.print(f"[yellow]Task {task.id} skipped: {task.result}[/yellow]")
                self._emit("task", f"Task {task.id} skipped: {task.result}")
            if not runnable:
                continue

            header = (
                f"Wave {index}/{len(waves)}: running {len(runnable)} task(s), "
                f"at most {limit} in parallel"
            )
            console.print(f"[dim]{header}[/dim]")
            self._emit("cycle", header)
            results = await self._run_wave(runnable, semaphore)
            for task, result in zip(runnable, results, strict=True):
                self._report_task(task, result)

    async def _run_wave(
        self,
        tasks: list[TaskModel],
        semaphore: asyncio.Semaphore,
    ) -> list[object]:
        """Run one wave of workers, bounded by ``semaphore``.

        ``return_exceptions=True`` keeps a single failing worker from cancelling
        its siblings; failures are classified by the caller.  Workers that are
        still queued when a stop is requested return :data:`_NOT_RUN` untouched.
        """

        async def _guarded(task: TaskModel) -> object:
            async with semaphore:
                if self._stop.is_set():
                    return _NOT_RUN
                worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
                return await worker.run()

        return await asyncio.gather(*(_guarded(t) for t in tasks), return_exceptions=True)

    def _partition(
        self,
        wave: list[TaskModel],
        tasks: list[TaskModel],
        forced: set[str] | None = None,
    ) -> tuple[list[TaskModel], list[tuple[TaskModel, str]]]:
        """Split ``wave`` into tasks that may run and tasks blocked by a dependency.

        Dependencies naming an unknown id are ignored (treated as satisfied).
        A dependency that failed, was skipped or never ran blocks the dependent
        task, which is then reported as skipped rather than run.  Ids in
        ``forced`` - the members of a broken dependency cycle - are exempt,
        otherwise the cycle could never be broken.
        """
        forced = forced or set()
        by_id: dict[str, TaskModel] = {}
        for task in tasks:
            key = _task_id(task)
            if key and key not in by_id:
                by_id[key] = task

        runnable: list[TaskModel] = []
        blocked: list[tuple[TaskModel, str]] = []
        for task in wave:
            if _task_id(task) in forced:
                runnable.append(task)
                continue
            reasons: list[str] = []
            for dep in _dependencies(task):
                target = by_id.get(dep)
                if target is None or target is task:
                    continue
                state = _status(target)
                if state != COMPLETED:
                    reasons.append(f"{dep} ({state})")
            if reasons:
                blocked.append((task, ", ".join(reasons)))
            else:
                runnable.append(task)
        return runnable, blocked

    def _report_task(self, task: TaskModel, result: object) -> None:
        """Classify one worker outcome and report it from the orchestrator.

        A worker that was queued behind a stop returns :data:`_NOT_RUN` and is
        recorded as ``skipped`` with :data:`SKIPPED_CANCELLED_PREFIX` - the run
        was cancelled, not blocked - which is what separates it from the
        dependency-blocked tasks marked in :meth:`_execute_plan`.
        """
        if result is _NOT_RUN:
            self._mark(task, SKIPPED, f"{SKIPPED_CANCELLED_PREFIX}stop requested")
            console.print(f"[yellow]Task {task.id} skipped: {task.result}[/yellow]")
            self._emit("task", f"Task {task.id} skipped: {task.result}")
        elif isinstance(result, asyncio.CancelledError):
            # A worker was cancelled: stop the loop and let cancellation surface.
            self._stop.set()
            raise result
        elif isinstance(result, BaseException):
            self._mark(task, FAILED, f"{type(result).__name__}: {result}")
            console.print(f"[red]Task {task.id} failed: {result}[/red]")
            self._emit("task", f"Task {task.id} failed: {_one_line(result)}")
        else:
            self._mark(task, COMPLETED, result)
            console.print(f"[green]Task {task.id} completed[/green]")
            self._emit("task", f"Task {task.id} completed: {_one_line(task.result)}")

    def _mark(self, task: TaskModel, status: TaskStatus, result: object) -> None:
        """Write a task's status/result and mirror the result into :attr:`results`."""
        task.status = status
        task.result = None if result is None else str(result)
        self.results[_task_id(task)] = task.result or ""

    def _skip_pending(self, wave: list[TaskModel], reason: str) -> None:
        """Mark every still-``PENDING`` task of *wave* as cancelled by the stop.

        This is the cancellation flavour of ``skipped``: the result carries
        :data:`SKIPPED_CANCELLED_PREFIX` followed by *reason*, which is what
        tells it apart from a task blocked by a failed dependency.  Already
        finished tasks are left alone.
        """
        for task in wave:
            if _status(task) == PENDING:
                self._mark(task, SKIPPED, f"{SKIPPED_CANCELLED_PREFIX}{reason}")

    # ------------------------------------------------------------------ tester

    async def _run_tester(self, tasks: list[TaskModel]) -> TestResult:
        """Run the test suite and let the tester agent judge it.

        A tester failure is turned into a failing :class:`TestResult` rather than
        an exception: a broken harness should feed the next cycle, not kill the
        run.  The tester is always asked about the immutable original goal, which
        is what defines success.
        """
        try:
            with self._spinner("Testing..."):
                tester = TesterAgent(client=self.client, settings=self.settings)
                raw = await tester.run(self.goal, tasks)
            return self._as_test_result(raw)
        except Exception as exc:  # noqa: BLE001 - a broken tester must not kill the run
            self.error = f"{type(exc).__name__}: {exc}"
            console.print(f"[red]Tester error: {self.error}[/red]")
            self._emit("error", f"Tester error: {self.error}")
            return TestResult(passed=False, summary=self.error, failures=[self.error])

    # ------------------------------------------------------------- state & IO

    def _get_tasks(self) -> list[TaskModel]:
        """The tasks of the current plan, empty before the first plan exists."""
        if self._current_plan is None:
            return []
        return list(self._current_plan.tasks)

    def _is_done(self) -> bool:
        """Whether the current plan proves the goal is met.

        Conservative on purpose, because this is the loop's exit predicate: an
        empty task list, a single non-completed task or a missing/failing test
        result all mean "not proven complete".  Only a real plan whose tasks all
        completed *and* a passing test run return ``True`` - never the
        ``0 >= 0`` accident that made the loop halt after one cycle.
        """
        if self._current_plan is None:
            return False
        tasks = self._get_tasks()
        if not tasks:
            return False
        if any(_status(task) != COMPLETED for task in tasks):
            return False
        return self.last_test_result is not None and self.last_test_result.passed

    def _build_focus(self) -> str:
        """Compose the planner prompt for the next cycle.

        :attr:`goal` stays immutable; the *focus* is the goal plus a concise
        digest of what earlier cycles already completed and which failures are
        still outstanding, so cycles build on each other instead of re-planning
        from scratch.  Both digests are truncated, so the prompt stays bounded
        even for a very long run.
        """
        parts = [f"Goal: {self.goal}"]
        completed = self._completed_summary()
        if completed:
            parts.append(
                "Already done in earlier cycles - build on it, do not redo it:\n" + completed
            )
        failures = self._outstanding_failures()
        if failures:
            parts.append("Outstanding test failures to fix in this cycle:\n" + failures)
        if self.history:
            parts.append(
                "The previous attempt did not fully succeed. Return a small plan that "
                "makes progress towards the goal above."
            )
        return "\n\n".join(parts)

    def _completed_summary(self) -> str:
        """One line per task completed in an earlier cycle, oldest first."""
        items: list[str] = []
        for index, plan in enumerate(self.history, start=1):
            for task in plan.tasks:
                if _status(task) != COMPLETED:
                    continue
                description = _one_line(task.description, 120)
                items.append(f"cycle {index} [{_task_id(task)}] {description}")
        if not items:
            return ""
        header = ""
        if len(items) > _MAX_HISTORY_ITEMS:
            omitted = len(items) - _MAX_HISTORY_ITEMS
            items = items[-_MAX_HISTORY_ITEMS:]
            header = f"({omitted} earlier completed task(s) omitted)\n"
        return header + "\n".join(f"- {item}" for item in items)

    def _outstanding_failures(self) -> str:
        """The failures of the last test run, capped so the prompt cannot explode."""
        result = self.last_test_result
        if result is None or result.passed or not result.failures:
            return ""
        failures = list(result.failures)
        text = "\n".join(f"- {_one_line(f, 300)}" for f in failures[-_MAX_FAILURE_ITEMS:])
        hidden = len(failures) - _MAX_FAILURE_ITEMS
        if hidden > 0:
            text += f"\n- ... and {hidden} more failure(s)"
        return text

    def _emit(self, event: str, message: str) -> None:
        """Forward an event to :attr:`on_event`.

        A broken event consumer must never be able to break the run, so the
        exception is swallowed - after a one-off warning - and the loop keeps
        going with the console as its only source of truth.
        """
        if self.on_event is None:
            return
        try:
            self.on_event(event, message)
        except Exception as exc:  # noqa: BLE001 - a consumer must not break the loop
            if not self._consumer_warned:
                self._consumer_warned = True
                console.print(
                    f"[dim]Event consumer raised {type(exc).__name__}: {_one_line(exc)}; "
                    "further event errors will be ignored.[/dim]"
                )

    def _halt(self, message: str) -> None:
        """Report a non-completion halt to the console and to event consumers.

        The event is named ``done`` because from the loop's own point of view the
        run is over either way, and the success path emits ``done`` the same
        way: a consumer therefore sees exactly one terminal ``done`` per run,
        with *message* carrying the reason.  A transport that adds a frame of
        its own must not reuse the name for its own terminal frame - the web
        layer relays this one as ``finished`` for exactly that reason.
        """
        self._done = False
        console.print(f"[bold yellow]{message}[/bold yellow]")
        self._emit("done", message)

    @contextlib.contextmanager
    def _spinner(self, message: str) -> Iterator[None]:
        """A Rich ``Status`` spinner on a terminal, a plain line when redirected.

        Rich live displays are only meaningful on a TTY; when output is piped or
        captured (CI, the web layer, log files) we degrade to one line so no
        escape codes leak.  Only ever used around a phase that prints nothing
        else, which keeps the console from interleaving.
        """
        if console.is_terminal:
            with Status(message, console=console):
                yield
        else:
            console.print(f"[dim]{message}[/dim]")
            yield

    # ------------------------------------------------------- normalisation

    @staticmethod
    def _as_plan(raw: object) -> Plan:
        """Coerce a planner result into a real :class:`Plan` of real tasks.

        The loop relies on owning mutable state, so whatever the planner returned
        is normalised into ``TaskModel`` objects whose ``status`` it can mutate.
        Anything that is not plan-shaped raises ``ValueError``, which the caller
        reports as a planner error.
        """
        if isinstance(raw, Plan):
            return raw
        if not hasattr(raw, "tasks"):
            raise ValueError(f"planner returned {raw!r}, which has no tasks")
        tasks = [
            task if isinstance(task, TaskModel) else Orchestrator._as_task(task, index)
            for index, task in enumerate(getattr(raw, "tasks", None) or [])
        ]
        return Plan(tasks=tasks, rationale=str(getattr(raw, "rationale", "") or "planner"))

    @staticmethod
    def _as_task(task: object, index: int) -> TaskModel:
        """Best-effort conversion of a duck-typed planner task into a TaskModel.

        ``TaskModel`` validates on construction, so a task missing an id, a
        description or carrying an unknown status degrades to a minimal valid
        task rather than failing the whole plan.
        """
        task_id = str(getattr(task, "id", "") or f"task-{index + 1}")
        try:
            return TaskModel(
                id=task_id,
                description=str(getattr(task, "description", "") or f"implement {task_id}"),
                files=[str(f) for f in (getattr(task, "files", None) or [])],
                dependencies=[str(d) for d in (getattr(task, "dependencies", None) or [])],
                status=TaskStatus(str(getattr(task, "status", PENDING) or PENDING)),
                result=getattr(task, "result", None),
            )
        except Exception:  # noqa: BLE001 - fall back to a minimal, valid task
            return TaskModel(id=task_id, description=f"implement {task_id}")

    @staticmethod
    def _as_test_result(raw: object) -> TestResult:
        """Coerce a tester result into a real :class:`TestResult`."""
        if isinstance(raw, TestResult):
            return raw
        return TestResult(
            passed=bool(getattr(raw, "passed", False)),
            summary=str(getattr(raw, "summary", "") or ""),
            failures=[str(f) for f in (getattr(raw, "failures", None) or [])],
        )

    @staticmethod
    def _plan_view(plan: Plan) -> Pretty:
        """A rich renderable for the plan panel."""
        dump = getattr(plan, "model_dump", None)
        if callable(dump):
            with contextlib.suppress(Exception):
                return Pretty(dump())
        return Pretty(plan)
