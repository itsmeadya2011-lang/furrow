"""Orchestrator: the planner -> worker -> tester loop with persistent state.

Historically this module held a stub ``_get_tasks()`` that returned ``[]``, a ``run()``
loop that broke after a single cycle, and a ``self.goal`` that was overwritten with the
latest failure list. That combination made the "infinite parallel development loop"
described in the README run exactly once and then print "Goal complete. Halting." even
when the tests had just failed.

The rewrite fixes that and the surrounding defects:

* ``self.tasks`` is the real, accumulated task state (replacing the no-op
  ``_get_tasks``), so cross-cycle statuses survive and ``_is_done()`` is meaningful.
* ``self.goal`` is never mutated; failure context lives in ``self._failure_context`` and
  is only ever appended to the *planning* goal, never the user's original requirement.
* ``asyncio.gather(..., return_exceptions=True)`` results are inspected with
  ``isinstance(result, BaseException)`` so a ``CancelledError`` (which is a
  ``BaseException`` in modern Python, not an ``Exception``) is recorded as a failure with
  a ``str`` result instead of being misclassified as success.
* ``settings.max_parallel_tasks`` is honoured via a semaphore, and
  :attr:`~furrow.models.TaskModel.dependencies` are honoured via wave scheduling.
* ``PlannerAgent.plan`` raising ``ValueError`` is caught and emitted as an ``ERROR``
  event instead of killing the run with a raw traceback.
* All output flows through :class:`~furrow.core.events.EventEmitter` so the web and CLI
  front-ends can observe a run.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, get_settings
from furrow.core.events import Event, EventEmitter, EventType
from furrow.jsonutil import truncate
from furrow.llm import LLMClient
from furrow.models import Plan, TaskModel, TaskStatus, TestResult

__all__ = [
    "DEFAULT_NO_PROGRESS_LIMIT",
    "MAX_GOAL_APPENDIX_CHARS",
    "Orchestrator",
    "RunSummary",
]

#: Consecutive cycles with no file writes after which the run stops to avoid spinning
#: forever rewriting an empty plan when the model cannot make progress.
DEFAULT_NO_PROGRESS_LIMIT = 3

#: Maximum number of characters the appended failure context may add to the planning goal.
#: Anything beyond this is truncated with :func:`furrow.jsonutil.truncate`.
MAX_GOAL_APPENDIX_CHARS = 4000


@dataclass
class RunSummary:
    """Terminal state of a finished :meth:`Orchestrator.run`."""

    #: The original goal, never overwritten.
    goal: str
    #: Number of cycles actually executed.
    cycles: int
    #: Tasks that completed successfully across the whole run.
    completed: int
    #: Tasks that failed across the whole run.
    failed: int
    #: Tasks skipped (blocked or superseded) across the whole run.
    skipped: int
    #: Why the loop stopped: ``"tests-passed"`` | ``"max-cycles"`` | ``"no-progress"`` |
    #: ``"planner-error"`` | ``"cancelled"``.
    stopped_because: str
    #: Every task the run ever knew about, with final statuses.
    tasks: list[TaskModel]

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dict view (tasks are serialised via ``model_dump(mode="json")``)."""
        return {
            "goal": self.goal,
            "cycles": self.cycles,
            "completed": self.completed,
            "failed": self.failed,
            "skipped": self.skipped,
            "stopped_because": self.stopped_because,
            "tasks": [task.model_dump(mode="json") for task in self.tasks],
        }


class Orchestrator:
    """Run the planner/worker/tester loop until a stop condition is met.

    Args:
        goal: The user's original requirement. Stored verbatim and never mutated;
            only the *planning* goal (see :meth:`_planning_goal`) ever carries
            accumulated failure context.
        client: Optional shared :class:`~furrow.llm.LLMClient`. Constructed lazily from
            ``settings`` when omitted.
        settings: Optional :class:`~furrow.config.Settings`. Defaults to the process-wide
            singleton via :func:`furrow.config.get_settings`.
        on_event: Optional callback invoked for every emitted event, used by the web/cli
            front-ends to observe a run.
        console: Optional rich :class:`~rich.console.Console` for live rendering. When
            omitted no console output is produced and the run is driven purely by events.
    """

    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        settings: Settings | None = None,
        *,
        on_event: Callable[[Event], None] | None = None,
        console: Console | None = None,
    ) -> None:
        """Bind the orchestrator to `goal` and lazily build its agents."""
        self.goal: str = goal
        self.settings: Settings = settings or get_settings()
        self.client: LLMClient = client or LLMClient(self.settings)
        self.console: Console | None = console
        self.emitter: EventEmitter = EventEmitter(
            on_event, console=console
        )
        self.planner: PlannerAgent = PlannerAgent(self.client, self.settings)
        # The tester is intentionally constructed once and reused across cycles: it is
        # stateless between runs and rebuilding it each cycle (as the old code did) was
        # pointless and hid the reuse contract.
        self.tester: TesterAgent = TesterAgent(self.client, self.settings)
        self.workspace: Path = self.settings.workspace_path()
        # Accumulated task state across cycles -- the real replacement for the old
        # ``_get_tasks()`` stub. New plans are deduped into here so a task's completed
        # status is not clobbered by a repeated planning cycle.
        self.tasks: list[TaskModel] = []
        self._failure_context: list[str] = []
        self._consecutive_no_progress: int = 0
        self._last_test_passed: bool = False
        self._cycle_count: int = 0
        self._cycle_writes: list[str] = []

    def _get_tasks(self) -> list[TaskModel]:
        """Return the accumulated tasks (backwards compatibility with the old API).

        The old implementation returned ``[]`` unconditionally, which is what made
        ``_is_done()`` see ``completed == 0`` and ``0 >= 0`` and halt the run after a
        single cycle. Returning the real list fixes that.
        """
        return self.tasks

    def _is_done(self) -> bool:
        """Return True when a natural terminal condition holds for the current run state.

        Terminal here means the loop's own stop conditions: ``max_cycles`` reached, the
        last test passed, or the no-progress limit expired. It never returns True solely
        because the task list is empty -- an empty list now means "the planner proposed
        nothing", which is the caller's signal rather than an automatic halt.
        """
        if self.settings.max_cycles and self._cycle_count >= self.settings.max_cycles:
            return True
        if self._last_test_passed:
            return True
        return self._consecutive_no_progress >= DEFAULT_NO_PROGRESS_LIMIT

    def _counts(self) -> tuple[int, int, int]:
        """Return ``(completed, failed, skipped)`` totals over ``self.tasks``."""
        completed = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in self.tasks if t.status == TaskStatus.FAILED)
        skipped = sum(1 for t in self.tasks if t.status == TaskStatus.SKIPPED)
        return completed, failed, skipped

    def _planning_goal(self) -> str:
        """The goal handed to the planner: original requirement + failure appendix.

        ``self.goal`` is never modified; the accumulated failure context from the most
        recent failing cycle is appended here only, capped at
        :data:`MAX_GOAL_APPENDIX_CHARS` via :func:`~furrow.jsonutil.truncate`. The
        original requirement text always leads the string, verbatim.
        """
        if not self._failure_context:
            return self.goal
        appendix = "\n".join(self._failure_context)
        combined = f"{self.goal}\n\n--- Failures from previous cycle ---\n{appendix}"
        if len(combined) > MAX_GOAL_APPENDIX_CHARS:
            combined = truncate(combined, MAX_GOAL_APPENDIX_CHARS)
        return combined

    def _record_failures(self, test_result: TestResult) -> None:
        """Replace the failure appendix with this cycle's failures and log it.

        The context is *replaced* each failing cycle (rather than appended) so the
        appendix never grows without bound across a long run.
        """
        self._failure_context = [
            truncate(failure, MAX_GOAL_APPENDIX_CHARS)
            for failure in test_result.failures
        ]
        self.emitter.emit(
            EventType.LOG,
            message=(
                f"recorded {len(self._failure_context)} failure(s) "
                "for the next planning cycle"
            ),
        )

    async def run(self) -> RunSummary:
        """Run the loop until a stop condition is met, emitting progress events.

        Emits ``RUN_START`` before the first cycle and ``RUN_END`` after the last,
        bracketing every cycle's ``cycle_start``/``cycle_end``. The returned
        :class:`RunSummary` carries the cumulative task totals and the reason the loop
        stopped.
        """
        # Reset all per-run state so the same instance can be run again without leaking
        # state from a previous run.
        self.tasks = []
        self._failure_context = []
        self._consecutive_no_progress = 0
        self._last_test_passed = False
        self._cycle_count = 0
        self._cycle_writes = []
        self.emitter.clear()
        self.emitter.current_cycle = 0
        completed, failed, skipped = self._counts()
        self.emitter.emit(
            EventType.RUN_START,
            goal=self.goal,
            workspace=str(self.workspace),
            provider=self.settings.provider,
            models=self.settings.models(),
            completed=completed,
            failed=failed,
            skipped=skipped,
        )
        stopped_because = "max-cycles"
        cycle = 0
        try:
            while True:
                if self.settings.max_cycles and cycle >= self.settings.max_cycles:
                    stopped_because = "max-cycles"
                    break
                cycle += 1
                self._cycle_count = cycle
                self.emitter.current_cycle = cycle
                self.emitter.emit(EventType.CYCLE_START, cycle=cycle)

                test_result = await self._cycle(cycle)
                if test_result is None:
                    # The planner raised; an ERROR event was already emitted in _cycle.
                    stopped_because = "planner-error"
                    break

                self._handle_test_result(test_result)

                completed, failed, skipped = self._counts()
                self.emitter.emit(
                    EventType.CYCLE_END,
                    cycle=cycle,
                    completed=completed,
                    failed=failed,
                    skipped=skipped,
                    passed=self._last_test_passed,
                )
                if self._last_test_passed:
                    stopped_because = "tests-passed"
                    break
                if self._consecutive_no_progress >= DEFAULT_NO_PROGRESS_LIMIT:
                    stopped_because = "no-progress"
                    break
        except asyncio.CancelledError:
            stopped_because = "cancelled"
            completed, failed, skipped = self._counts()
            self.emitter.emit(
                EventType.RUN_END,
                cycles=cycle,
                completed=completed,
                failed=failed,
                skipped=skipped,
                stopped_because=stopped_because,
            )
            raise
        except Exception as exc:  # noqa: BLE001 - report and re-raise; front-ends present
            self.emitter.emit(
                EventType.ERROR,
                cycle=cycle,
                stage="run",
                message=str(exc),
            )
            completed, failed, skipped = self._counts()
            self.emitter.emit(
                EventType.RUN_END,
                cycles=cycle,
                completed=completed,
                failed=failed,
                skipped=skipped,
                stopped_because="error",
            )
            raise
        completed, failed, skipped = self._counts()
        self.emitter.emit(
            EventType.RUN_END,
            cycles=cycle,
            completed=completed,
            failed=failed,
            skipped=skipped,
            stopped_because=stopped_because,
        )
        return RunSummary(
            goal=self.goal,
            cycles=cycle,
            completed=completed,
            failed=failed,
            skipped=skipped,
            stopped_because=stopped_because,
            tasks=self.tasks,
        )

    async def _cycle(self, cycle: int) -> TestResult | None:
        """Plan, execute and test one cycle.

        Returns the :class:`~furrow.models.TestResult`, or ``None`` when the planner
        could not produce a plan (a ``ValueError`` is logged as an ERROR event rather than
        re-raised, so the run stops cleanly instead of with a raw traceback).
        """
        try:
            plan: Plan = await self.planner.plan(self._planning_goal())
        except ValueError as exc:
            self.emitter.emit(
                EventType.ERROR,
                cycle=cycle,
                stage="plan",
                message=str(exc),
            )
            return None
        self.emitter.emit(
            EventType.PLAN,
            cycle=cycle,
            rationale=plan.rationale,
            task_count=len(plan.tasks),
            tasks=[
                {
                    "id": task.id,
                    "description": task.description,
                    "files": list(task.files),
                    "dependencies": list(task.dependencies),
                    "status": task.status.value,
                }
                for task in plan.tasks
            ],
        )
        batch = self._register_tasks(plan)
        executed = await self._execute_batch(cycle, batch)
        test_result = await self.tester.run(self.goal, executed)
        self.emitter.emit(
            EventType.TEST,
            cycle=cycle,
            passed=test_result.passed,
            skipped=test_result.skipped,
            summary=test_result.summary,
            failures=list(test_result.failures),
        )
        return test_result

    def _handle_test_result(self, test_result: TestResult) -> None:
        """Update run state from this cycle's test outcome.

        ``skipped`` is neutral -- nothing ran -- so it neither stops the loop nor counts
        as a pass. ``passed`` stops the loop with ``"tests-passed"``. Anything else has its
        failures recorded so the next planning cycle can attack them.
        """
        if test_result.skipped:
            self.emitter.emit(
                EventType.LOG,
                message="no runnable test suite detected; continuing without a pass",
            )
            self._last_test_passed = False
            self._consecutive_no_progress = (
                0 if self._cycle_writes else self._consecutive_no_progress + 1
            )
            return
        if test_result.passed:
            self._last_test_passed = True
            self._consecutive_no_progress = 0
            return
        self._last_test_passed = False
        self._record_failures(test_result)
        self._consecutive_no_progress = (
            0 if self._cycle_writes else self._consecutive_no_progress + 1
        )

    def _register_tasks(self, plan: Plan) -> list[TaskModel]:
        """Merge ``plan.tasks`` into ``self.tasks`` and return the newly added ones.

        Tasks are deduped by ``(id, description)``: a task id that re-appears in a later
        cycle (with the same description) is left alone so its already-completed status
        and result are not clobbered by a fresh pending copy.
        """
        existing = {(t.id, t.description) for t in self.tasks}
        new_tasks: list[TaskModel] = []
        for task in plan.tasks:
            key = (task.id, task.description)
            if key in existing:
                continue
            self.tasks.append(task)
            new_tasks.append(task)
            existing.add(key)
        return new_tasks

    async def _execute_batch(
        self, cycle: int, tasks: list[TaskModel]
    ) -> list[TaskModel]:
        """Run ``tasks`` honouring :attr:`~furrow.models.TaskModel.dependencies`.

        Tasks are executed in dependency-respecting waves: each wave contains every
        not-yet-run task whose dependencies are all ``COMPLETED``, and a wave is not
        launched until the previous one finishes, so dependent tasks never overlap.
        Concurrency within a wave is bounded by
        ``settings.max_parallel_tasks`` via a semaphore.

        A task whose dependency already ``FAILED`` or was ``SKIPPED`` is marked
        ``SKIPPED`` (with a ``result`` naming the blocker) and never run. A dependency
        cycle among the planner's output is detected when a full wave produces no progress
        and is resolved by marking the remaining tasks ``SKIPPED`` -- the wave count is
        also hard-capped so the loop can never hang.
        """
        self._cycle_writes = []
        executed: list[TaskModel] = []
        pending = [t for t in tasks if t.status == TaskStatus.PENDING]
        # Upper bound on waves: N tasks can form at most N-1 dependency edges in a chain,
        # so after N+1 waves with no progress something is stale (a cycle) and we stop.
        max_waves = len(tasks) + 1
        semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
        while pending:
            max_waves -= 1
            if max_waves < 0:
                for task in pending:
                    self._skip_task(task, cycle, "unresolvable dependency cycle")
                    executed.append(task)
                pending.clear()
                break
            completed_ids = {
                t.id for t in self.tasks if t.status == TaskStatus.COMPLETED
            }
            failed_ids = {
                t.id
                for t in self.tasks
                if t.status in (TaskStatus.FAILED, TaskStatus.SKIPPED)
            }
            # First pass: skip anything whose dependency already failed / was skipped.
            for task in list(pending):
                stalled = set(task.dependencies) & failed_ids
                if stalled:
                    blocker = sorted(stalled)[0]
                    self._skip_task(
                        task,
                        cycle,
                        f"dependency {blocker!r} did not complete "
                        f"({self._status_of(blocker)})",
                    )
                    executed.append(task)
                    pending.remove(task)
            # Second pass: collect tasks whose dependencies are all completed.
            ready = [
                t
                for t in pending
                if set(t.dependencies) <= completed_ids
            ]
            if not ready:
                # No ready and no newly-skippable tasks => remaining tasks are in a cycle
                # (their dependencies are neither completed nor failed/skipped, yet they
                # can never become runnable).
                for task in pending:
                    self._skip_task(task, cycle, "unresolvable dependency cycle")
                    executed.append(task)
                pending.clear()
                break
            for task in ready:
                pending.remove(task)
            results = await asyncio.gather(
                *(self._run_task(cycle, task, semaphore) for task in ready),
                return_exceptions=True,
            )
            for task, result in zip(ready, results, strict=True):
                written: list[str] = []
                if isinstance(result, BaseException):
                    # Note BaseException, not Exception: CancelledError is a
                    # BaseException in modern Python and must not be misclassified as
                    # success. The result is always stored as a str, never the exception.
                    task.status = TaskStatus.FAILED
                    task.result = truncate(str(result), 2000)
                else:
                    summary, written = result
                    task.status = TaskStatus.COMPLETED
                    task.result = summary
                    self._cycle_writes.extend(written)
                self.emitter.emit(
                    EventType.TASK_END,
                    cycle=cycle,
                    task_id=task.id,
                    description=task.description,
                    status=task.status.value,
                    written_files=written,
                    result=task.result,
                )
                executed.append(task)
        return executed

    async def _run_task(
        self, cycle: int, task: TaskModel, semaphore: asyncio.Semaphore
    ) -> tuple[str, list[str]]:
        """Run one worker for ``task`` under ``semaphore``; return ``(summary, written)``.

        The worker's ``written_files`` are returned -- not stashed on ``self`` -- because
        several tasks in a wave run concurrently and a shared attribute would race. Raises
        propagate back to :meth:`_execute_batch`'s ``gather`` so they are recorded as a
        per-task failure rather than aborting the whole batch. The task is left in the
        ``RUNNING`` state until :meth:`_execute_batch` resolves the ``gather`` result.
        """
        async with semaphore:
            task.status = TaskStatus.RUNNING
            self.emitter.emit(
                EventType.TASK_START,
                cycle=cycle,
                task_id=task.id,
                description=task.description,
                files=list(task.files),
                dependencies=list(task.dependencies),
            )
            worker = WorkerAgent(task=task, client=self.client, settings=self.settings)
            summary = await worker.run()
            return summary, list(worker.written_files)

    def _skip_task(self, task: TaskModel, cycle: int, reason: str) -> None:
        """Mark ``task`` SKIPPED with ``result=reason`` and emit its TASK_END event."""
        task.status = TaskStatus.SKIPPED
        task.result = reason
        self.emitter.emit(
            EventType.TASK_END,
            cycle=cycle,
            task_id=task.id,
            description=task.description,
            status=task.status.value,
            written_files=[],
            result=reason,
        )

    def _status_of(self, task_id: str) -> str:
        """Return the status string of a previously-known task, or ``"unknown"``."""
        for task in self.tasks:
            if task.id == task_id:
                return task.status.value
        return "unknown"
