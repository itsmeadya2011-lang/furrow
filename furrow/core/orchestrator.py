"""The Furrow loop: plan, execute, test, repeat — until the goal is done.

The orchestrator is the only piece of Furrow that owns cross-cycle state: the
current plan, the per-cycle history, the stall counter, the repair context and
the original goal (which is never rewritten). It drives the three agents and
publishes every step of its progress on an :class:`~furrow.core.events.EventBus`
instead of printing, so the terminal reporter, the web UI and the tests all
observe exactly the same stream.

Termination is a deliberate decision, not a side effect of counting task
statuses: a run ends when the planner says the goal is complete, when the
cycle budget is spent, when a caller asks it to stop, or when it has stalled.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent, WorkerResult
from furrow.config import Plan, TaskModel, TaskStatus, TestResult
from furrow.config import settings as default_settings
from furrow.core.events import (
    EventBus,
    cycle_completed,
    cycle_started,
    log,
    plan_created,
    run_completed,
    run_failed,
    run_started,
    task_completed,
    task_failed,
    task_skipped,
    task_started,
    tests_completed,
)
from furrow.core.reporting import RichReporter
from furrow.errors import LLMError, StructuredOutputError
from furrow.llm import LLMClient
from furrow.tools import WorkspaceTools

if TYPE_CHECKING:
    from furrow.config import Settings

#: Reason reported when the planner returns an empty plan: the only genuine
#: "the work is finished" signal the loop has.
EMPTY_PLAN_REASON = "planner reported the goal complete"

#: Repair context is capped so a long run cannot grow an unbounded prompt. The
#: most recent slice is kept: the planner has already had the older failures.
MAX_REPAIR_CONTEXT_CHARS = 4_000

#: Cap on the per-task failure note handed to a worker retry.
MAX_TASK_FEEDBACK_CHARS = 2_000

#: Heading inserted between the goal and the accumulated repair context.
REPAIR_CONTEXT_HEADER = "## Failures reported by previous cycles (fix these first)"

#: Floor for the stall budget, so a narrow `max_parallel_tasks` still gets a
#: few chances before the run gives up.
MIN_STALL_CYCLES = 3


@dataclass(slots=True)
class CycleRecord:
    """One entry of the run history: what a single cycle actually achieved."""

    cycle: int
    planned_tasks: int = 0
    completed: int = 0
    failed: int = 0
    skipped: int = 0
    tests_passed: bool = False
    tests_summary: str = ""
    planner_error: str = ""
    empty_plan: bool = False
    stopped: bool = False

    @property
    def progressed(self) -> bool:
        """Whether this cycle moved the goal forward."""
        return self.completed > 0 or self.tests_passed

    def to_dict(self) -> dict[str, object]:
        """JSON-safe view, for the web UI and for assertions in tests."""
        return {
            "cycle": self.cycle,
            "planned_tasks": self.planned_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "tests_passed": self.tests_passed,
            "tests_summary": self.tests_summary,
            "planner_error": self.planner_error,
            "empty_plan": self.empty_plan,
            "stopped": self.stopped,
            "progressed": self.progressed,
        }


@dataclass(slots=True)
class _Outcome:
    """The loop's view of a finished worker, reduced to what it acts on."""

    ok: bool
    summary: str = ""
    error: str | None = None
    duration: float = 0.0


@dataclass(slots=True)
class _ExecutionOutcome:
    """Tally of one cycle's task execution."""

    completed: int = 0
    failed: int = 0
    skipped: int = 0


def _normalise_worker_result(result: WorkerResult, duration: float) -> _Outcome:
    """Project a :class:`~furrow.agents.worker.WorkerResult` onto an `_Outcome`.

    The parameter is the real, runtime-imported `WorkerResult` and the three
    fields it needs are read as attributes rather than through `getattr`. That
    is deliberate: a renamed or misspelled field then raises `AttributeError`
    here — at import time for a static checker, immediately at runtime — instead
    of silently defaulting to `ok=False` and quietly failing every task in the
    run. The loop only needs `ok`/`summary`/`error`, so nothing else from the
    worker crosses this boundary.
    """
    return _Outcome(
        ok=result.ok,
        summary=result.summary or "",
        error=result.error,
        duration=duration,
    )


class Orchestrator:
    """Run the plan → execute → test loop for a goal until it terminates.

    State lives on the instance, so a run has memory between cycles: the plan
    produced by the planner, the tasks' statuses, the history of every cycle,
    the accumulated repair context and the stall counter.
    """

    def __init__(
        self,
        goal: str,
        *,
        client: LLMClient | None = None,
        bus: EventBus | None = None,
        settings: Settings | None = None,
    ) -> None:
        # The goal the user actually asked for. Never reassigned: test
        # failures add context, they never rewrite the intent.
        self._original_goal = goal
        self._current_goal = goal
        self._settings: Settings = settings if settings is not None else default_settings

        if bus is not None:
            self._bus = bus
            self._reporter: RichReporter | None = None
        else:
            self._bus = EventBus()
            # No caller-supplied bus: attach the terminal reporter so a plain
            # `Orchestrator(goal).run()` still shows progress. A caller that
            # owns its bus (the web server) renders the same stream itself.
            self._reporter = RichReporter(self._bus)

        self._client = client
        self._owns_client = False
        self._tools: WorkspaceTools | None = None

        self._plan: Plan | None = None
        self._history: list[CycleRecord] = []
        self._cycles = 0
        self._stall_cycles = 0
        self._stop_requested = False
        self._stop_reason = ""
        self._repair_context = ""
        self._task_feedback: dict[str, str] = {}
        self._closed = False

    # ── Public surface ───────────────────────────────────────────────────

    @property
    def cycles(self) -> int:
        """Number of cycles started so far."""
        return self._cycles

    @property
    def plan(self) -> Plan | None:
        """The most recent plan, or `None` before the first planning pass."""
        return self._plan

    @property
    def stopped(self) -> bool:
        """Whether `request_stop` has been called."""
        return self._stop_requested

    @property
    def history(self) -> list[CycleRecord]:
        """A snapshot of the per-cycle records collected so far."""
        return list(self._history)

    @property
    def goal(self) -> str:
        """The goal as it will be handed to the planner next: the original
        goal plus any accumulated repair context."""
        return self._current_goal

    @property
    def original_goal(self) -> str:
        """The user's goal, verbatim, for the whole life of the run."""
        return self._original_goal

    @property
    def repair_context(self) -> str:
        """The bounded block of recent failures appended to the goal."""
        return self._repair_context

    @property
    def bus(self) -> EventBus:
        """The event bus every progress update is published on."""
        return self._bus

    @property
    def client(self) -> LLMClient | None:
        """The shared LLM client, or `None` until one is first needed."""
        return self._client

    def request_stop(self, reason: str = "") -> None:
        """Ask the loop to finish after the current cycle.

        Safe to call from another task (a signal handler, a web socket, a
        keyboard interrupt wrapper). A non-empty reason is what the final
        `run_completed` event reports.
        """
        self._stop_requested = True
        if reason:
            self._stop_reason = reason

    async def aclose(self) -> None:
        """Release the reporter and any LLM client this instance created.

        Idempotent, and safe when nothing was ever created: a caller-supplied
        client belongs to the caller and is left alone.
        """
        if self._closed:
            return
        self._closed = True
        reporter, self._reporter = self._reporter, None
        if reporter is not None:
            reporter.close()
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    # ── The loop ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Loop plan → execute → test until a termination condition fires."""
        await self._bus.emit(
            run_started(self._original_goal, max_cycles=self._settings.max_cycles)
        )
        reason = "loop finished"
        try:
            while True:
                if self._stop_requested:
                    reason = self._stop_reason or "stop requested"
                    break
                if self._cycle_budget_reached():
                    reason = f"reached the max_cycles budget of {self._max_cycles}"
                    break

                self._cycles += 1
                await self._bus.emit(cycle_started(self._cycles))
                await self._cycle()

                # An empty plan is the planner saying "there is nothing left to
                # do". That — not a task count — is what a completed goal means.
                if self._plan is not None and self._plan.is_empty:
                    reason = EMPTY_PLAN_REASON
                    break
                if self._is_done():
                    reason = self._termination_reason()
                    break

                # Give other tasks (a web socket, a signal handler) a turn so
                # a stop request requested during the cycle is seen promptly.
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            await self._bus.emit(run_failed("cancelled"))
            raise
        except Exception as exc:
            await self._bus.emit(run_failed(f"{type(exc).__name__}: {exc}"))
            raise
        await self._bus.emit(run_completed(self._cycles, reason))

    async def _cycle(self) -> CycleRecord:
        """Run one plan → execute → test pass and record what it achieved."""
        cycle = self._cycles
        self._current_goal = self._planning_goal()
        record = CycleRecord(cycle=cycle)

        planner = PlannerAgent(client=self._ensure_client(), settings=self._settings)
        try:
            plan = await planner.plan(self._current_goal)
        except (StructuredOutputError, LLMError) as exc:
            # An unparseable plan must never surface as a traceback out of
            # run(). It counts as a stalled cycle and the stall rule decides
            # when the run gives up.
            record.planner_error = str(exc)
            await self._bus.emit(
                log(f"cycle {cycle}: planning failed: {exc}", "error", cycle=cycle)
            )
            self._stall_cycles += 1
            self._history.append(record)
            await self._bus.emit(cycle_completed(cycle, False))
            return record

        self._plan = plan
        record.planned_tasks = len(plan.tasks)
        await self._bus.emit(plan_created(plan.model_dump(mode="json"), cycle=cycle))
        duplicates = plan.duplicate_ids()
        if duplicates:
            await self._bus.emit(
                log(
                    f"cycle {cycle}: plan reuses task id(s): {', '.join(duplicates)}",
                    "warning",
                    cycle=cycle,
                )
            )

        if plan.is_empty:
            record.empty_plan = True
            self._history.append(record)
            await self._bus.emit(log(EMPTY_PLAN_REASON, "info", cycle=cycle))
            await self._bus.emit(cycle_completed(cycle, True))
            return record

        outcome = await self._execute_plan(plan)
        record.completed = outcome.completed
        record.failed = outcome.failed
        record.skipped = outcome.skipped
        if self._stop_requested:
            # Stopping mid-plan: leave the workspace alone rather than starting
            # a fresh test run on half-finished work.
            record.stopped = True
            self._history.append(record)
            await self._bus.emit(cycle_completed(cycle, False))
            return record

        test_result = await TesterAgent(
            client=self._ensure_client(),
            settings=self._settings,
            workspace=self._settings.workspace,
        ).run(self._current_goal, plan.tasks)
        await self._bus.emit(
            tests_completed(
                test_result.passed,
                test_result.summary,
                list(test_result.failures),
                cycle=cycle,
            )
        )

        record.tests_passed = test_result.passed
        record.tests_summary = test_result.summary
        if test_result.passed:
            self._repair_context = ""
        else:
            self._append_repair_context(test_result, plan)
        self._update_progress(newly_completed=outcome.completed, tests_passed=test_result.passed)

        self._history.append(record)
        await self._bus.emit(cycle_completed(cycle, test_result.passed))
        return record

    # ── Termination ──────────────────────────────────────────────────────

    def _is_done(self) -> bool:
        """Whether the run must stop.

        Four independent conditions, and nothing else. In particular task
        statuses are *not* consulted: a cycle in which every planned task
        completed is normal mid-run state — the next planning pass is what
        decides whether the goal is actually finished, and until it returns an
        empty plan the loop keeps going.
        """
        if self._stop_requested:
            return True
        if self._cycle_budget_reached():
            return True
        if self._plan is not None and self._plan.is_empty:
            return True
        return self._stall_cycles >= self._stall_threshold

    @property
    def _max_cycles(self) -> int:
        """Cycle budget; `0` means unlimited."""
        return self._settings.max_cycles

    def _cycle_budget_reached(self) -> bool:
        """Whether the configured cycle budget has been used up."""
        return self._max_cycles > 0 and self._cycles >= self._max_cycles

    @property
    def _stall_threshold(self) -> int:
        """Consecutive fruitless cycles tolerated before giving up.

        `max(3, max_parallel_tasks)`: a wrong plan usually fails the same way
        several times before it is visibly hopeless, and a run that can execute
        N tasks in parallel tends to re-plan N times on the same bad idea, so
        the budget is scaled by the width of the machine. The floor of 3 keeps
        a narrow configuration from giving up after a single misstep, while
        still bounding the cost of an "infinite" loop that is going nowhere.
        """
        return max(MIN_STALL_CYCLES, self._settings.max_parallel_tasks)

    def _update_progress(self, *, newly_completed: int, tests_passed: bool) -> None:
        """Reset the stall counter on real progress, otherwise extend it."""
        if newly_completed > 0 or tests_passed:
            self._stall_cycles = 0
        else:
            self._stall_cycles += 1

    def _termination_reason(self) -> str:
        """Human-readable reason for stopping, matching `_is_done`."""
        if self._stop_requested:
            return self._stop_reason or "stop requested"
        if self._cycle_budget_reached():
            return f"reached the max_cycles budget of {self._max_cycles}"
        if self._plan is not None and self._plan.is_empty:
            return EMPTY_PLAN_REASON
        return (
            f"stalled: {self._stall_cycles} consecutive cycle(s) completed no new task "
            f"and the tests never passed (threshold {self._stall_threshold})"
        )

    # ── Task execution ───────────────────────────────────────────────────

    async def _execute_plan(self, plan: Plan) -> _ExecutionOutcome:
        """Execute every task of ``plan`` in dependency/file-safe batches."""
        batches = await self._execution_batches(plan.tasks)
        limit = max(1, self._settings.max_parallel_tasks)
        semaphore = asyncio.Semaphore(limit)
        outcome = _ExecutionOutcome()
        # task id -> why it will never produce usable work (failed or skipped)
        blocked: dict[str, str] = {}
        # files an earlier failure left unfinished
        blocked_files: set[str] = set()

        for batch in batches:
            runnable: list[TaskModel] = []
            for task in batch:
                reason = self._skip_reason(task, blocked, blocked_files)
                if reason is not None or self._stop_requested:
                    text = (
                        "stop requested before this task started"
                        if self._stop_requested
                        else str(reason)
                    )
                    await self._mark_skipped(task, text)
                    outcome.skipped += 1
                    blocked[task.id] = text
                else:
                    runnable.append(task)
            if not runnable:
                continue

            # return_exceptions=True: one exploding task must not take its
            # siblings down with it.
            results = await asyncio.gather(
                *(self._run_task(task, semaphore) for task in runnable),
                return_exceptions=True,
            )
            for task, result in zip(runnable, results, strict=True):
                if isinstance(result, asyncio.CancelledError):
                    raise result
                if isinstance(result, BaseException):
                    failure = _Outcome(
                        ok=False,
                        error=f"{type(result).__name__}: {result}",
                    )
                else:
                    failure = result
                if await self._finish_task(task, failure):
                    outcome.completed += 1
                else:
                    outcome.failed += 1
                    blocked[task.id] = task.error or "failed"
                    blocked_files.update(task.files)
        return outcome

    async def _run_task(
        self, task: TaskModel, semaphore: asyncio.Semaphore
    ) -> _Outcome:
        """Run one worker under the parallelism limit and time it.

        The semaphore is what turns `max_parallel_tasks` into a real bound: a
        whole batch may be gathered at once, but never more than ``limit``
        workers are in flight.
        """
        async with semaphore:
            task.attempts += 1
            task.status = TaskStatus.IN_PROGRESS
            task.started_at = time.time()
            await self._bus.emit(
                task_started(task.id, task.description, list(task.files), cycle=self._cycles)
            )
            agent = WorkerAgent(
                task,
                self._ensure_client(),
                settings=self._settings,
                tools=self._ensure_tools(),
                attempt=task.attempts,
                feedback=self._feedback_for(task),
            )
            started = time.monotonic()
            try:
                result = await agent.run()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Defensive: the worker's contract says run() never raises, but
                # a broken worker must fail its own task only.
                return _Outcome(
                    ok=False,
                    error=f"{type(exc).__name__}: {exc}",
                    duration=time.monotonic() - started,
                )
            return _normalise_worker_result(result, time.monotonic() - started)

    async def _finish_task(self, task: TaskModel, outcome: _Outcome) -> bool:
        """Record a finished task, emit its event, return True if it succeeded."""
        task.finished_at = time.time()
        if outcome.ok:
            task.status = TaskStatus.COMPLETED
            task.result = outcome.summary
            task.error = None
            self._task_feedback.pop(task.id, None)
            await self._bus.emit(
                task_completed(task.id, outcome.summary, outcome.duration, cycle=self._cycles)
            )
            return True
        task.status = TaskStatus.FAILED
        task.error = outcome.error or "worker reported failure without detail"
        task.result = outcome.summary or None
        self._task_feedback[task.id] = self._failure_note(task)
        await self._bus.emit(
            task_failed(task.id, task.error, outcome.duration, cycle=self._cycles)
        )
        return False

    async def _mark_skipped(self, task: TaskModel, reason: str) -> None:
        """Mark a task that could not run and announce it."""
        task.status = TaskStatus.SKIPPED
        task.error = reason
        task.finished_at = time.time()
        await self._bus.emit(task_skipped(task.id, reason, cycle=self._cycles))

    def _skip_reason(
        self, task: TaskModel, blocked: dict[str, str], blocked_files: set[str]
    ) -> str | None:
        """Why ``task`` must not run yet, or `None` if it may run.

        A task is skipped when something it needed did not happen: a dependency
        that failed (transitively — skipped dependencies are blocked too), or a
        file an earlier failure left unfinished. Continuing in either case would
        compound a broken state; the next cycle re-plans from scratch anyway.
        """
        unmet = [dep for dep in task.dependencies if dep in blocked]
        if unmet:
            return f"blocked by unfinished dependency: {', '.join(sorted(unmet))}"
        conflicts = sorted(set(task.files) & blocked_files)
        if conflicts:
            return f"files left unfinished by an earlier failure: {', '.join(conflicts)}"
        return None

    async def _execution_batches(self, tasks: list[TaskModel]) -> list[list[TaskModel]]:
        """Group tasks into batches that are safe to run concurrently.

        ``dependencies`` are hard ordering constraints and declared ``files``
        are soft mutual exclusion: a file claimed by one batch is unavailable to
        every later batch, so two tasks that would edit the same file are
        serialised even without a dependency edge. Everything else is packed
        greedily into the same batch, so parallelism is as wide as the plan
        allows.

        The planner decycles its output, but malformed input must not hang the
        loop or produce an unsafe batch: if nothing can be scheduled — a
        dependency cycle, or a self/duplicate reference — the first remaining
        task is emitted alone, which cannot place it beside anything it depends
        on and guarantees the loop makes progress.
        """
        known = {task.id for task in tasks}
        remaining = list(tasks)
        scheduled: set[str] = set()
        claimed_files: set[str] = set()
        batches: list[list[TaskModel]] = []

        while remaining:
            batch: list[TaskModel] = []
            batch_ids: set[str] = set()
            batch_files: set[str] = set()
            deferred: list[TaskModel] = []
            for task in remaining:
                files = set(task.files)
                # Unknown ids are ignored (they can never be scheduled) and a
                # self-reference is not an edge; a duplicate id is held back so
                # two records for one id never run at the same time. The id must
                # be checked against `batch_ids` as well as `scheduled`: a
                # duplicate sitting next to its twin in `remaining` has not been
                # added to `scheduled` yet, so checking only `scheduled` would
                # put both records for one id in the same batch.
                unmet = {
                    dep
                    for dep in task.dependencies
                    if dep != task.id and dep in known and dep not in scheduled
                }
                if (
                    unmet
                    or task.id in scheduled
                    or task.id in batch_ids
                    or files & (claimed_files | batch_files)
                ):
                    deferred.append(task)
                    continue
                batch.append(task)
                batch_ids.add(task.id)
                batch_files |= files

            if not batch:
                blocked = remaining[0]
                batches.append([blocked])
                scheduled.add(blocked.id)
                claimed_files.update(blocked.files)
                remaining = remaining[1:]
                await self._bus.emit(
                    log(
                        f"task {blocked.id} scheduled alone: the plan contains a "
                        f"dependency cycle or unresolvable references",
                        "warning",
                        cycle=self._cycles,
                    )
                )
                continue

            batches.append(batch)
            scheduled |= batch_ids
            claimed_files.update(batch_files)
            remaining = deferred
        return batches

    # ── Goal and repair context ──────────────────────────────────────────

    def _planning_goal(self) -> str:
        """The goal handed to the planner: the original, plus repair context."""
        if not self._repair_context:
            return self._original_goal
        return f"{self._original_goal}\n\n{REPAIR_CONTEXT_HEADER}\n{self._repair_context}"

    def _append_repair_context(self, test_result: TestResult, plan: Plan) -> None:
        """Accumulate the latest failures into a bounded repair context.

        The block is capped at :data:`MAX_REPAIR_CONTEXT_CHARS` and the oldest
        text is dropped, so the planner prompt cannot grow without limit. The
        original goal is never touched — this context is appended to it.
        """
        lines = [f"Test run for cycle {self._cycles}: {test_result.summary}"]
        lines.extend(f"- {failure}" for failure in test_result.failures)
        for task in plan.tasks:
            if task.status is TaskStatus.FAILED:
                lines.append(f"- task {task.id} failed: {task.error or 'no detail given'}")
        block = "\n".join(lines)
        combined = f"{self._repair_context}\n{block}" if self._repair_context else block
        if len(combined) > MAX_REPAIR_CONTEXT_CHARS:
            combined = "[earlier context truncated]\n" + combined[-MAX_REPAIR_CONTEXT_CHARS:]
        self._repair_context = combined

    def _failure_note(self, task: TaskModel) -> str:
        """Feedback text handed to the worker when a task is retried."""
        parts = [part for part in (task.error, task.result) if part]
        return "\n".join(parts)[:MAX_TASK_FEEDBACK_CHARS]

    def _feedback_for(self, task: TaskModel) -> str:
        """Worker feedback for ``task``, empty when it has never failed."""
        note = self._task_feedback.get(task.id)
        if not note:
            return ""
        return f"A previous attempt at this task failed:\n{note}"

    # ── Lazily built collaborators ───────────────────────────────────────

    def _ensure_client(self) -> LLMClient:
        """Build the shared LLM client on first use, never with a positional None."""
        if self._client is None:
            self._client = LLMClient(settings=self._settings)
            self._owns_client = True
        return self._client

    def _ensure_tools(self) -> WorkspaceTools:
        """Build the workspace sandbox once and reuse it across workers."""
        if self._tools is None:
            self._tools = WorkspaceTools.from_settings(self._settings)
        return self._tools

    # ── State accessors used by tests ────────────────────────────────────

    def _get_tasks(self) -> list[TaskModel]:
        """Return a copy of the current plan's task list, or ``[]`` before the first plan.

        This is a deliberate test seam, not a production accessor: the run
        loop's termination logic in :meth:`_is_done` deliberately does *not*
        consult task statuses — counting completed tasks as a stop signal is
        exactly the ``0 >= 0`` false-positive that once made the loop declare
        victory on cycle one before any work existed. The plan's task list is
        still useful for *inspection* after a run, so this method exposes it to
        tests without wiring production code to a status-counting heuristic.

        ``tests/test_orchestrator.py`` asserts on this directly (lines 443, 447,
        696); removing it would break those tests. Returns a shallow copy so
        callers cannot mutate the live plan.
        """
        if self._plan is None:
            return []
        return list(self._plan.tasks)
