from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from typing import Any

from rich.console import Console
from rich.status import Status

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.state import CycleRecord, RunState, StateStore
from furrow.llm import LLMClient

EventCallback = Callable[[str, dict[str, Any]], None]

#: How many ``CycleRecord``s the persisted state file keeps.
#:
#: ``max_cycles = 0`` means "loop forever" (the whole point of the agent), so an
#: append-only history would grow the state file without bound. 50 recent
#: cycles is plenty for a human reading the file at the end of a run -- the arc
#: of the run, not every iteration of it -- while keeping the file small. The
#: OLDEST records are the ones dropped, because the tail of a log is the part a
#: reader actually wants. Must stay >= 1.
MAX_HISTORY_RECORDS = 50

#: Error text given to tasks that a requested stop left undispatched.
STOPPED_BEFORE_START = "Run stopped before this task started"

#: Error text given to tasks abandoned by the stalled-wave invariant guard.
STALLED_WAVE = "Wave made no progress; task was never dispatched"

#: ``done`` reason for a run that verified its work and can stop.
SUCCESS_REASON = "All tasks completed and tests passed"


class PlannerUnavailableError(RuntimeError):
    """The loop stopped because the planner never produced a single plan.

    Raised out of :meth:`Orchestrator.run` *after* the terminal ``done`` event,
    carrying that event's reason. It exists because a terminal reason alone is
    not enough for the callers that never see the event stream: ``furrow start``
    passes no ``on_event`` and exits ``0`` unless the run raises, so without this
    a run in which the planner failed on every cycle -- no task planned, no task
    run, nothing verified -- would exit ``0`` in silence. Every other exit stays
    a plain return: reaching ``max_cycles`` or the no-progress guard is a
    decision the user asked for, not a failure of the loop to do its job.
    """


console = Console()


class Orchestrator:
    def __init__(
        self,
        goal: str,
        client: LLMClient | None = None,
        on_event: EventCallback | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.goal = goal
        self.focus = goal
        self.client = client or LLMClient()
        self.settings = settings or Settings()
        self.planner = PlannerAgent(client=self.client, settings=self.settings)
        self.tester = TesterAgent(client=self.client, settings=self.settings)
        self.tasks: list[TaskModel] = []
        self.cycles: int = 0
        self._running = False
        self._stopped = False
        self._on_event = on_event
        self._last_test_result: TestResult | None = None
        self._failure_history: list[str] = []
        self._history: list[CycleRecord] = []
        self._planner_error: str | None = None
        self._planner_failures: int = 0
        self._store = StateStore(self.settings)

    async def run(self) -> None:
        self._running = True
        try:
            while True:
                # Hand control back to the event loop once per cycle. A cycle in
                # which the planner, every task and the tester all fail
                # immediately never suspends on its own, so without this the
                # loop spins without ever yielding -- and a coroutine that never
                # yields cannot be cancelled, timed out or stopped: ``stop()``,
                # ``Ctrl+C`` and ``asyncio.wait_for`` would all have nothing to
                # interrupt. Yielding costs one event-loop tick per cycle and
                # makes every cycle boundary a point the run can be ended at.
                await asyncio.sleep(0)
                if self._stopped:
                    self._finish("Stop requested")
                    break
                self.cycles += 1
                self._emit("cycle_start", {"cycle": self.cycles})
                await self._cycle()
                if self._stopped:
                    self._finish("Stop requested")
                    break
                if self._is_done():
                    self._store.clear()
                    self._finish(SUCCESS_REASON)
                    break
                reason = self._stop_reason()
                if reason is not None:
                    self._finish(reason)
                    if self._planner_is_giving_up():
                        raise PlannerUnavailableError(reason)
                    break
        except asyncio.CancelledError:
            self._finish("Cancelled")
            raise
        finally:
            self._running = False

    def _finish(self, reason: str) -> None:
        """Emit the terminal ``done`` event for this run."""
        self._emit("done", {"reason": reason, "cycles": self.cycles})

    def stop(self) -> None:
        self._stopped = True

    @property
    def is_running(self) -> bool:
        return self._running

    async def _cycle(self) -> None:
        self._planner_error = None
        plan = await self._plan()
        self._emit(
            "plan",
            {
                "rationale": plan.rationale,
                "task_count": len(plan.tasks),
                "tasks": [
                    {
                        "id": t.id,
                        "description": t.description,
                        "files": t.files,
                        "dependencies": t.dependencies,
                        "status": t.status,
                    }
                    for t in plan.tasks
                ],
            },
        )
        self.tasks = plan.tasks
        if self.tasks:
            await self._execute_waves()
        test_result = await self._test()
        self._last_test_result = test_result
        self._emit(
            "test_result",
            {
                "passed": test_result.passed,
                "summary": test_result.summary,
                "failures": test_result.failures,
            },
        )
        if test_result.passed:
            console.print(
                f"[green]Tests passed: {test_result.summary}[/green]"
            )
        else:
            console.print(
                f"[red]Tests failed: {test_result.summary}[/red]"
            )
            for failure in test_result.failures:
                console.print(f"  • {failure}")
            self.focus = "\n".join(test_result.failures)
        completed_ids = [t.id for t in self.tasks if t.status == TaskStatus.COMPLETED]
        failed_ids = [t.id for t in self.tasks if t.status == TaskStatus.FAILED]
        self._emit(
            "cycle_end",
            {
                "cycle": self.cycles,
                "passed": test_result.passed,
                "completed": completed_ids,
                "failed": failed_ids,
            },
        )
        self._store.save(
            RunState(
                goal=self.goal,
                cycles=self.cycles,
                tasks=self.tasks,
                history=self._record_history(test_result, completed_ids, failed_ids),
            )
        )
        self._update_failure_history(test_result)

    def _record_history(
        self,
        test_result: TestResult,
        completed_ids: list[str],
        failed_ids: list[str],
    ) -> list[CycleRecord]:
        """Append this cycle to the run history and return it, oldest first.

        ``RunState(history=[...])`` REPLACES the list, so a fresh single-element
        list silently dropped every earlier cycle and the state file only ever
        described the last one. The history therefore lives on the instance and
        each cycle appends to it, trimmed to the most recent
        :data:`MAX_HISTORY_RECORDS` entries (the oldest go first).
        """
        self._history.append(
            CycleRecord(
                cycle=self.cycles,
                passed=test_result.passed,
                summary=test_result.summary,
                failures=test_result.failures,
                task_ids=[t.id for t in self.tasks],
                completed=completed_ids,
                failed=failed_ids,
            )
        )
        if len(self._history) > MAX_HISTORY_RECORDS:
            del self._history[: len(self._history) - MAX_HISTORY_RECORDS]
        return list(self._history)

    async def _plan(self) -> Plan:
        with Status("[bold yellow]Planning...", console=console):
            try:
                plan = await self.planner.plan(self.goal, context=self.focus)
            except Exception as e:  # noqa: BLE001
                # The fallback plan is empty, so an empty plan cannot on its own
                # tell "the planner says there is nothing to do" apart from "the
                # planner failed and we never found out". Record which one it was:
                # only the former may end the run as a success.
                self._planner_error = str(e)
                self._planner_failures += 1
                self._emit(
                    "error",
                    {
                        "message": str(e),
                        "where": "planner",
                    },
                )
                return Plan()
        self._planner_failures = 0
        return plan

    async def _test(self) -> TestResult:
        with Status("[bold yellow]Testing...", console=console):
            try:
                return await self.tester.run(self.goal, self.tasks)
            except Exception as e:  # noqa: BLE001
                self._emit(
                    "error",
                    {
                        "message": str(e),
                        "where": "tester",
                    },
                )
                return TestResult(passed=False, summary=str(e), failures=[str(e)])

    async def _execute_waves(self) -> None:
        self._mark_blocked()
        ready = self._ready_tasks()
        semaphore = asyncio.Semaphore(max(1, self.settings.max_parallel_tasks))
        with Status("[bold yellow]Executing tasks...", console=console):
            while ready:
                # A stop has to end the wave loop, not just the tasks already in
                # flight: ``run_task`` returns early for everything it has not
                # started, those tasks stay PENDING, and ``_ready_tasks`` would
                # hand the very same list back on the next pass -- the loop that
                # burned CPU forever and kept ``run()`` from ever returning.
                if self._stopped:
                    self._close_out_unstarted(STOPPED_BEFORE_START)
                    return
                wave = ready
                ready = []
                before = self._status_snapshot()

                async def run_task(task: TaskModel) -> None:
                    async with semaphore:
                        if self._stopped:
                            return
                        self._emit(
                            "task_start",
                            {
                                "id": task.id,
                                "description": task.description,
                            },
                        )
                        task.status = TaskStatus.RUNNING
                        try:
                            await self._make_worker(task).run()
                            if task.status == TaskStatus.RUNNING:
                                task.status = TaskStatus.COMPLETED
                                task.result = task.result or ""
                        except Exception as e:  # noqa: BLE001
                            # ``WorkerAgent.run`` owns ``task.attempts``: it
                            # counts the attempt before it starts, so the
                            # orchestrator must not count it again.
                            self._record_failure(task, str(e))
                        self._emit(
                            "task_done",
                            {
                                "id": task.id,
                                "status": task.status,
                                "result": task.result,
                                "error": task.error,
                            },
                        )

                coros = [asyncio.create_task(run_task(t)) for t in wave]
                await asyncio.gather(*coros, return_exceptions=True)
                for task, coro in zip(wave, coros):
                    # ``run_task`` settles and reports the tasks it managed to
                    # start. Anything it could not (a coroutine cancelled or
                    # blown up before it marked the task) has to be closed out
                    # here, or it stays PENDING and comes back as ready work.
                    if task.status not in (TaskStatus.PENDING, TaskStatus.RUNNING):
                        continue
                    if coro.cancelled():
                        self._record_failure(task, "Task execution was cancelled")
                    elif coro.exception() is not None:
                        self._record_failure(task, str(coro.exception()))
                if self._stopped:
                    self._close_out_unstarted(STOPPED_BEFORE_START)
                    return
                ready = self._ready_tasks()
                if ready and self._status_snapshot() == before:
                    # Forward-progress guard. Running a wave always moves its
                    # tasks off PENDING, so a wave that changed no status while
                    # ``_ready_tasks`` still reports work is an invariant
                    # violation, not a reason to keep looping: the next pass
                    # would see exactly the same list and spin. Close the tasks
                    # out and leave the loop -- returning here matters, because
                    # ``ready`` still holds the un-dispatchable work.
                    self._close_out_unstarted(STALLED_WAVE)
                    return
        # Reached here only by falling out of the loop, i.e. with no ready work
        # left. The stop and stalled-wave paths return from inside it and never
        # get this far, so a task the *run* declined to dispatch is still closed
        # out as SKIPPED rather than judged against its dependencies.
        self._close_out_stranded()
        self._propagate_blocked()

    def _status_snapshot(self) -> tuple[TaskStatus, ...]:
        """Every task status, in plan order (what "did this wave change?" reads)."""
        return tuple(task.status for task in self.tasks)

    def _record_failure(self, task: TaskModel, error: str) -> None:
        """Leave a task FAILED, escalating to SKIPPED once its budget is spent."""
        task.status = TaskStatus.FAILED
        task.error = error
        # ``WorkerAgent.run`` owns the attempt counter, so this never increments.
        if task.attempts >= self.settings.max_task_attempts:
            task.status = TaskStatus.SKIPPED
            task.error = (
                f"Exceeded max_task_attempts "
                f"({self.settings.max_task_attempts}): "
                f"{task.error}"
            )

    def _close_out_unstarted(self, reason: str) -> None:
        """Give planned-but-never-dispatched work a terminal status and a reason.

        Used for the two ways the *run* can end a wave loop with tasks still
        ``PENDING``: a requested stop, and the stalled-wave invariant guard.
        ``PENDING`` is the input to ``_ready_tasks``, so leaving it behind is
        both misleading and -- via that loop -- a hang. These tasks are marked
        ``SKIPPED``: ``BLOCKED`` would blame the plan for work the run itself
        declined to start, and ``SKIPPED`` already means "deliberately not run"
        for the attempt-budget escalation. Keeping it distinct from a real
        failure is what the ``attempts`` counter is for, and it stays at 0 here.
        Each one gets a ``task_done`` event so the event stream stays truthful;
        on the stalled path that event supersedes the one ``run_task`` already
        emitted, so the last word on a task's status is always its terminal one.

        Both paths return straight out of :meth:`_execute_waves`, so they also
        run *before* :meth:`_close_out_stranded` gets a chance to blame a
        dependency: a task stranded by a stop is the run's decision to report
        as ``SKIPPED``, not the plan's.
        """
        for task in self.tasks:
            if task.status != TaskStatus.PENDING:
                continue
            self._close_out(task, TaskStatus.SKIPPED, reason)

    def _close_out_stranded(self) -> None:
        """Block every task that can never run because a dependency died.

        ``_ready_tasks`` dispatches only work whose dependencies are ``COMPLETED``
        and ``_propagate_blocked`` follows only ``BLOCKED`` ones, so a task whose
        dependency ended ``failed`` or ``skipped`` satisfied neither: it was never
        ready and never blocked, and it left the run ``pending`` with
        ``error=None``, no ``task_done`` event and nothing at all to explain
        where it went. It is not pending work, it is work the plan can never
        satisfy, so it is closed out as ``BLOCKED`` -- the same verdict a task
        blocked by a dangling, self-referential or cyclic dependency already
        gets -- with an ``error`` naming the dependency and the state it died in.

        Runs to a fixpoint so a dependant listed *before* the task that stranded
        it is closed out too, and the caller follows it with
        :meth:`_propagate_blocked`, which carries the blocking on to whatever
        depends on these tasks using the wording the rest of the blocked chain
        uses. A dependency that is already ``BLOCKED`` is deliberately left to
        that pass, so one blocked chain keeps one wording.
        """
        task_map = {t.id: t for t in self.tasks}
        # The dead ends ``_propagate_blocked`` does not already own: BLOCKED is
        # its case, and a COMPLETED dependency needs no closing out at all.
        dead_ends = (TaskStatus.FAILED, TaskStatus.SKIPPED)
        changed = True
        while changed:
            changed = False
            for task in self.tasks:
                if task.status != TaskStatus.PENDING:
                    continue
                for dep in task.dependencies:
                    dep_task = task_map.get(dep)
                    if dep_task is None or dep_task.status not in dead_ends:
                        # Unknown, still working, completed, or already handled
                        # by the blocked pass: not this method's business.
                        continue
                    self._close_out(
                        task,
                        TaskStatus.BLOCKED,
                        f"Dependency {dep!r} ended {dep_task.status.value} "
                        "and can never complete",
                    )
                    changed = True
                    break

    def _close_out(self, task: TaskModel, status: TaskStatus, error: str) -> None:
        """Put a task in a terminal state with a reason, and report it as done."""
        task.status = status
        task.error = error
        self._emit(
            "task_done",
            {
                "id": task.id,
                "status": task.status,
                "result": task.result,
                "error": task.error,
            },
        )

    def _make_worker(self, task: TaskModel) -> WorkerAgent:
        return WorkerAgent(task=task, client=self.client, settings=self.settings)

    def _mark_blocked(self) -> None:
        ids = {t.id for t in self.tasks}
        for task in self.tasks:
            for dep in task.dependencies:
                if dep not in ids:
                    task.status = TaskStatus.BLOCKED
                    task.error = f"Dependency {dep!r} does not exist in plan"
                    break
                if dep == task.id:
                    task.status = TaskStatus.BLOCKED
                    task.error = "Self-dependency detected"
                    break
        self._detect_cycles()
        self._propagate_blocked()

    def _detect_cycles(self) -> None:
        """Mark every task that sits on a dependency cycle ``BLOCKED``.

        Invariant this implementation exists to hold: **a task coloured ``GRAY``
        is always on the DFS stack.**  The colouring is derived from the stack
        itself, so the two can never disagree, which is what makes the back-edge
        lookup below safe.

        The previous recursive DFS broke that invariant by ``return``-ing out of
        the middle of a walk (after finding a cycle, or after noticing that
        *some* task anywhere was ``BLOCKED``) without writing ``BLACK``, leaving
        those nodes ``GRAY`` forever.  A later node depending on one of them then
        read ``color[dep] == GRAY``, called ``path.index(dep)`` on a stack that
        did not contain it, and raised ``ValueError`` straight out of ``run()``:
        no ``error`` event, no ``done`` event, whole process dead.  Members of a
        cycle are exactly the tasks between the back edge's target and the end of
        the stack, so nothing needs to be guessed and nothing needs to be
        abandoned mid-walk.

        Iterative rather than recursive so a long dependency chain cannot hit
        the interpreter's recursion limit either.  Every task that merely
        *depends* on a cycle member is closed out by
        :meth:`_propagate_blocked`, which runs to a fixpoint.
        """
        task_map = {t.id: t for t in self.tasks}
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {t.id: WHITE for t in self.tasks}

        def block_cycle(start_index: int, stack: list[tuple[str, Iterator[str]]]) -> None:
            """Block every task from ``start_index`` to the top of ``stack``."""
            for member_id, _ in stack[start_index:]:
                member = task_map.get(member_id)
                if member is not None and member.status != TaskStatus.BLOCKED:
                    member.status = TaskStatus.BLOCKED
                    member.error = "Dependency cycle detected"

        for root in self.tasks:
            if color[root.id] != WHITE:
                continue
            root_task = task_map.get(root.id)
            color[root.id] = GRAY
            stack: list[tuple[str, Iterator[str]]] = [
                (root.id, iter(root_task.dependencies if root_task else []))
            ]
            while stack:
                node_id, pending = stack[-1]
                descended = False
                for dep in pending:
                    if dep not in color:
                        # Unknown dependency: _mark_blocked already closed it out.
                        continue
                    state = color[dep]
                    if state == GRAY:
                        # Back edge: the whole walk is on the stack, so this
                        # index lookup cannot raise.
                        for index, (framed_id, _) in enumerate(stack):
                            if framed_id == dep:
                                block_cycle(index, stack)
                                break
                    elif state == WHITE:
                        dep_task = task_map.get(dep)
                        color[dep] = GRAY
                        stack.append((dep, iter(dep_task.dependencies if dep_task else [])))
                        descended = True
                        break
                if descended:
                    continue
                stack.pop()
                color[node_id] = BLACK

    def _propagate_blocked(self) -> None:
        task_map = {t.id: t for t in self.tasks}
        changed = True
        while changed:
            changed = False
            for task in self.tasks:
                if task.status == TaskStatus.BLOCKED:
                    continue
                for dep in task.dependencies:
                    dep_task = task_map.get(dep)
                    if dep_task and dep_task.status == TaskStatus.BLOCKED:
                        task.status = TaskStatus.BLOCKED
                        task.error = f"Dependency {dep!r} is blocked"
                        changed = True
                        break

    def _ready_tasks(self) -> list[TaskModel]:
        completed = {t.id for t in self.tasks if t.status == TaskStatus.COMPLETED}
        return [
            t
            for t in self.tasks
            if t.status == TaskStatus.PENDING
            and all(dep in completed for dep in t.dependencies)
        ]

    def _is_done(self) -> bool:
        if not self.tasks:
            # An empty plan is only "there is nothing left to do" when the
            # planner actually said so. ``_plan`` falls back to an empty plan
            # when the planner raises, and those two plans are identical objects,
            # so without this check a run that never planned a single task
            # finished with the success message having verified nothing at all.
            return self._planner_error is None and not self._failure_history
        return all(
            t.status == TaskStatus.COMPLETED for t in self.tasks
        ) and (
            self._last_test_result is not None
            and self._last_test_result.passed
        )

    def _should_continue(self) -> bool:
        return self._stop_reason() is None

    def _stop_reason(self) -> str | None:
        """Why the loop must stop now, or ``None`` to run another cycle."""
        # Before the cycle bound, and not folded into the no-progress guard: a
        # run whose planner never produced a plan did not really get N cycles
        # of work done, so "Reached max_cycles" would misdescribe it, and the
        # no-progress guard cannot see it at all -- a passing verdict clears
        # that history, so the signature it compares never repeats.
        planner_reason = self._planner_failure_reason()
        if planner_reason is not None:
            return planner_reason
        max_cycles = self.settings.max_cycles
        if max_cycles > 0 and self.cycles >= max_cycles:
            return f"Reached max_cycles ({max_cycles})"
        if self._is_no_progress():
            return self._no_progress_reason()
        return None

    def _planner_is_giving_up(self) -> bool:
        """True when every cycle so far lost its plan to a planner failure.

        Bounded by ``max_consecutive_failures``, the same knob that bounds a run
        making no progress: one transient planner failure is retried (the client
        has already spent its own retries), a permanently broken planner is not
        worth an unbounded number of cycles, and ``0`` disables the guard exactly
        as it disables the failure-signature guard.
        """
        threshold = self.settings.max_consecutive_failures
        return (
            self._planner_error is not None
            and threshold > 0
            and self._planner_failures >= threshold
        )

    def _planner_failure_reason(self) -> str | None:
        if not self._planner_is_giving_up():
            return None
        cycles = "cycle" if self._planner_failures == 1 else "cycles"
        return (
            f"Planner failed to produce a plan for {self._planner_failures} "
            f"consecutive {cycles} (last error: {self._planner_error})"
        )

    def _is_no_progress(self) -> bool:
        threshold = self.settings.max_consecutive_failures
        if threshold <= 0 or len(self._failure_history) < threshold:
            return False
        recent = self._failure_history[-threshold:]
        return len(set(recent)) == 1

    def _no_progress_reason(self) -> str:
        return (
            f"No progress after {self.settings.max_consecutive_failures} "
            f"identical failing cycles"
        )

    def _update_failure_history(self, test_result: TestResult) -> None:
        if test_result.passed:
            self._failure_history.clear()
            return
        failed_ids = sorted(
            t.id for t in self.tasks if t.status == TaskStatus.FAILED
        )
        signature = "|".join(
            sorted(test_result.failures) + [f"task:{tid}" for tid in failed_ids]
        )
        self._failure_history.append(signature)

    def _failure_signature(self) -> str:
        if not self._last_test_result:
            return ""
        failed_ids = sorted(
            t.id for t in self.tasks if t.status == TaskStatus.FAILED
        )
        return "|".join(
            sorted(self._last_test_result.failures)
            + [f"task:{tid}" for tid in failed_ids]
        )

    def _emit(self, name: str, payload: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(name, payload)
        except Exception:  # noqa: BLE001,S110
            pass
