"""Contracts for :mod:`furrow.core.orchestrator`.

The orchestrator is the only place the plan -> execute -> test loop exists, and it
is the module that shipped with ``_get_tasks() -> []``: a stub that made every
cycle report "Goal complete" after the plan step, so the product did nothing at
all while exiting cleanly. These tests pin the loop's observable behaviour with
the three agents replaced by stubs:

* the task list the loop reports is the one the planner produced;
* the goal is never rewritten -- previous test failures travel in
  ``plan_context`` instead;
* the loop always terminates -- ``max_cycles``, ``fail_fast_cycles`` (counting
  hard failures *and* red test runs), the dependency scheduler, and the error
  boundary all have to bound it;
* worker concurrency respects ``max_parallel_tasks`` and dependency order.

Every ``run()`` is wrapped in :func:`asyncio.wait_for` so a loop that stops
terminating fails the test instead of hanging the suite.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.worker import WorkerResult
from furrow.config import Plan, TaskModel, TaskStatus, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import MAX_CONSECUTIVE_CYCLE_FAILURES, Orchestrator
from tests.conftest import Settings, make_settings

#: A task spec is either a bare id (no dependencies) or an ``(id, deps)`` pair.
TaskSpec = str | tuple[str, Sequence[str]]

#: Upper bound on any ``Orchestrator.run()`` below, so a non-terminating loop
#: raises ``TimeoutError`` instead of hanging the suite.
RUN_TIMEOUT = 30.0

GREEN = TestResult(passed=True, summary="all green")
RED = TestResult(passed=False, summary="1 failed", failures=["tests/test_widget.py::test_build"])


# --------------------------------------------------------------------------
# Recording + stubs
# --------------------------------------------------------------------------


def _plan(specs: Sequence[TaskSpec]) -> Plan:
    """A fresh :class:`Plan`; fresh every time because ``_cycle`` mutates status."""
    tasks = [
        TaskModel(id=spec, description=f"task {spec}")
        if isinstance(spec, str)
        else TaskModel(id=spec[0], description=f"task {spec[0]}", dependencies=list(spec[1]))
        for spec in specs
    ]
    return Plan(tasks=tasks, rationale="stubbed plan")


class _Recorder:
    """Everything the stubbed agents saw, plus worker overlap in real time."""

    def __init__(
        self, specs: Sequence[Sequence[TaskSpec]] = (), *, repeat_last: bool = True
    ) -> None:
        #: One entry per planned cycle; the last one repeats unless told otherwise.
        self.specs: list[list[TaskSpec]] = [list(entry) for entry in specs]
        self.repeat_last = repeat_last
        #: ``(goal, failure_context)`` per planner call, in order.
        self.planner_calls: list[tuple[str, str | None]] = []
        #: ``(goal, [task ids])`` per tester call, in order.
        self.tester_calls: list[tuple[str, list[str]]] = []
        self.worker_calls: list[str] = []
        self.entered: list[tuple[str, float]] = []
        self.exited: list[tuple[str, float]] = []
        self.active = 0
        self.peak = 0

    def next_plan(self) -> Plan:
        index = len(self.planner_calls) - 1
        if index < len(self.specs):
            return _plan(self.specs[index])
        if self.repeat_last and self.specs:
            return _plan(self.specs[-1])
        raise AssertionError(f"planner call #{index + 1} has no scripted plan")

    def enter(self, task_id: str) -> None:
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.entered.append((task_id, time.monotonic()))

    def leave(self, task_id: str) -> None:
        self.active -= 1
        self.exited.append((task_id, time.monotonic()))

    def entered_at(self, task_id: str) -> float:
        return next(when for name, when in self.entered if name == task_id)

    def exited_at(self, task_id: str) -> float:
        return next(when for name, when in self.exited if name == task_id)


def _Planner(recorder: _Recorder) -> type:
    """A stub planner whose ``plan`` takes ``(goal, failure_context)``.

    Two positional parameters on the bound method is what the *real*
    :class:`~furrow.agents.planner.PlannerAgent` exposes, so
    ``_planner_accepts_context()`` probes this stub exactly as it probes production.
    """

    class _StubPlanner:
        def __init__(self, *, client: Any = None, settings: Settings | None = None) -> None:
            self.client = client
            self.settings = settings

        async def plan(self, goal: str, failure_context: str | None = None) -> Plan:
            recorder.planner_calls.append((goal, failure_context))
            return recorder.next_plan()

    return _StubPlanner


def _InlinePlanner(recorder: _Recorder) -> type:
    """A stub planner whose ``plan`` takes only ``goal``.

    The orchestrator must not call this with two arguments; it has to splice the
    previous failures into the goal instead.
    """

    class _StubPlanner:
        def __init__(self, *, client: Any = None, settings: Settings | None = None) -> None:
            self.client = client
            self.settings = settings

        async def plan(self, goal: str) -> Plan:
            recorder.planner_calls.append((goal, None))
            return recorder.next_plan()

    return _StubPlanner


def _WorkerStub(recorder: _Recorder, run_task: Callable[..., Any]) -> type:
    class _StubWorker:
        def __init__(
            self, *, task: TaskModel, client: Any = None, settings: Settings | None = None
        ) -> None:
            self.task = task

        async def run(self) -> Any:
            recorder.worker_calls.append(self.task.id)
            recorder.enter(self.task.id)
            try:
                return await run_task(self.task, recorder)
            finally:
                recorder.leave(self.task.id)

    return _StubWorker


def _TesterStub(recorder: _Recorder, run_tests: Callable[..., Any]) -> type:
    class _StubTester:
        def __init__(self, *, client: Any = None, settings: Settings | None = None) -> None:
            self.client = client
            self.settings = settings

        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            recorder.tester_calls.append((goal, [task.id for task in tasks]))
            return await run_tests(goal, tasks, recorder)

    return _StubTester


# --------------------------------------------------------------------------
# Default collaborators + the build helper every test uses
# --------------------------------------------------------------------------


async def _ok_worker(task: TaskModel, rec: _Recorder) -> WorkerResult:
    return WorkerResult(success=True, summary=f"{task.id} implemented")


async def _failing_worker(task: TaskModel, rec: _Recorder) -> WorkerResult:
    return WorkerResult(success=False, summary="", error="worker could not finish")


async def _passing_tester(goal: str, tasks: list[TaskModel], rec: _Recorder) -> TestResult:
    return GREEN


async def _failing_tester(goal: str, tasks: list[TaskModel], rec: _Recorder) -> TestResult:
    return RED


async def _exploding_tester(goal: str, tasks: list[TaskModel], rec: _Recorder) -> TestResult:
    raise RuntimeError("tester crashed")


async def _yielding_worker(task: TaskModel, rec: _Recorder) -> WorkerResult:
    """A worker that stays in flight long enough for its peers to overlap it."""
    for _ in range(5):
        await asyncio.sleep(0)
    return WorkerResult(success=True, summary=f"{task.id} implemented")


def _scripted_tester(*results: TestResult) -> Callable[..., Any]:
    """A tester returning ``results`` in order; an extra call fails the test."""

    async def run(goal: str, tasks: list[TaskModel], rec: _Recorder) -> TestResult:
        index = len(rec.tester_calls) - 1
        if index >= len(results):
            raise AssertionError(f"tester call #{index + 1} has no scripted result")
        return results[index]

    return run


def _build(
    workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
    *,
    specs: Sequence[Sequence[TaskSpec]] = (("1", ()),),
    worker: Callable[..., Any] = _ok_worker,
    tester: Callable[..., Any] = _passing_tester,
    inline_planner: bool = False,
    goal: str = "ship the widget",
    **settings_overrides: Any,
) -> tuple[_Recorder, Orchestrator]:
    """An :class:`Orchestrator` with all three collaborators replaced."""
    recorder = _Recorder(specs)
    factory = _InlinePlanner if inline_planner else _Planner
    monkeypatch.setattr(orchestrator_module, "PlannerAgent", factory(recorder))
    monkeypatch.setattr(orchestrator_module, "TesterAgent", _TesterStub(recorder, tester))
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", _WorkerStub(recorder, worker))
    settings = make_settings(workspace=workspace.workspace, **settings_overrides)
    return recorder, Orchestrator(goal, client=object(), settings=settings)


def _tasks(*statuses: str) -> list[TaskModel]:
    return [
        TaskModel(id=f"t{index}", description="d", status=status)
        for index, status in enumerate(statuses, start=1)
    ]


# --------------------------------------------------------------------------
# The headline regression
# --------------------------------------------------------------------------


async def test_get_tasks_is_not_a_stub(
    tmp_workspace: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One cycle must surface the planner's tasks, not an empty list.

    This is the bug that shipped: ``_get_tasks()`` returned ``[]``, so
    ``_is_done()`` compared ``0 >= 0`` and every run printed "Goal complete" after
    the plan step without executing or testing anything.
    """
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ()), ("2", ("1",))]],
        max_cycles=1,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    tasks = o._get_tasks()
    assert len(tasks) > 0
    assert [task.id for task in tasks] == ["1", "2"]
    assert tasks is o.current_tasks
    # The plan was not merely parsed -- it was executed and tested.
    assert rec.worker_calls == ["1", "2"]
    assert [task.status for task in tasks] == [TaskStatus.COMPLETED] * 2
    assert rec.tester_calls == [("ship the widget", ["1", "2"])]
    assert o.cycles == 1


# --------------------------------------------------------------------------
# _is_done
# --------------------------------------------------------------------------


def test_is_done_is_false_while_work_remains(tmp_workspace: Settings) -> None:
    o = Orchestrator("g", client=object(), settings=tmp_workspace)
    o.current_tasks = _tasks("pending", "running", "completed")
    o._tests_passed = True
    assert o._is_done() is False


def test_is_done_is_true_only_when_every_task_completed(tmp_workspace: Settings) -> None:
    o = Orchestrator("g", client=object(), settings=tmp_workspace)
    o.current_tasks = _tasks("completed", "completed")
    o._tests_passed = True
    assert o._is_done() is True


def test_is_done_is_false_when_any_task_failed(tmp_workspace: Settings) -> None:
    """A green suite does not rescue a task that failed to complete."""
    o = Orchestrator("g", client=object(), settings=tmp_workspace)
    o.current_tasks = _tasks("completed", "failed")
    o._tests_passed = True
    assert o._is_done() is False


def test_is_done_is_false_over_a_red_test_run(tmp_workspace: Settings) -> None:
    """Otherwise the loop reports "Goal complete" while the suite is failing."""
    o = Orchestrator("g", client=object(), settings=tmp_workspace)
    o.current_tasks = _tasks("completed", "completed")
    o._tests_passed = False
    assert o._is_done() is False


def test_is_done_is_false_after_a_hard_failure(tmp_workspace: Settings) -> None:
    o = Orchestrator("g", client=object(), settings=tmp_workspace)
    o.current_tasks = _tasks("completed")
    o._tests_passed = True
    o._hard_failure = True
    assert o._is_done() is False


def test_is_done_is_true_when_no_tests_ran(tmp_workspace: Settings) -> None:
    """A cycle that planned nothing ran no tests, so it is complete."""
    o = Orchestrator("g", client=object(), settings=tmp_workspace)
    o.current_tasks = []
    o._tests_passed = None
    assert o._is_done() is True


# --------------------------------------------------------------------------
# The goal must survive every cycle
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "inline_planner", [False, True], ids=["context_argument", "goal_spliced_inline"]
)
async def test_goal_is_never_overwritten(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
    inline_planner: bool,
) -> None:
    """Failures travel forward in ``plan_context``; ``goal`` is read-only.

    Feeding the previous test failures back by overwriting the goal made every
    cycle forget the original objective. Both planner shapes are covered: one
    that accepts a context argument and one that does not.
    """
    goal = "ship the widget"
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ())], [("1", ())]],
        tester=_failing_tester,
        inline_planner=inline_planner,
        goal=goal,
        max_cycles=1,
        fail_fast_cycles=50,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert o.cycles == 1
    assert o.goal == o.original_goal == goal
    assert o.plan_context
    assert "tests/test_widget.py::test_build" in o.plan_context
    # The first cycle has nothing to feed forward, so no context is passed at all.
    assert rec.planner_calls[0] == (goal, None)

    context = o.plan_context
    o.cycles = 2
    await asyncio.wait_for(o._cycle(), timeout=RUN_TIMEOUT)

    second_goal, second_context = rec.planner_calls[1]
    if inline_planner:
        assert second_context is None
        assert second_goal == f"{goal}\n\n{context}"
    else:
        assert second_goal == goal
        assert second_context == context
    assert o.goal == o.original_goal == goal
    assert o.plan_context


async def test_a_passing_cycle_clears_the_planning_context(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Green tests drop the accumulated context, so cycle 3 plans from the goal."""
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ())]],
        tester=_scripted_tester(RED, GREEN, GREEN),
        max_cycles=1,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    context = o.plan_context
    assert context
    assert o.goal == o.original_goal

    # Cycle 2 still sees cycle 1's failures, then clears the context on green.
    o.cycles = 2
    await asyncio.wait_for(o._cycle(), timeout=RUN_TIMEOUT)
    assert rec.planner_calls[1] == (o.goal, context)
    assert o.plan_context is None

    # Cycle 3 therefore gets no context at all.
    o.cycles = 3
    await asyncio.wait_for(o._cycle(), timeout=RUN_TIMEOUT)
    assert rec.planner_calls[2] == (o.goal, None)
    assert o.goal == o.original_goal


# --------------------------------------------------------------------------
# Loop bounds
# --------------------------------------------------------------------------


async def test_max_cycles_is_honoured_exactly(
    tmp_workspace: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``max_cycles=2`` runs two cycles and no more.

    The WORKER is the failing part on purpose: with a green worker *and* a green
    tester the goal is genuinely complete after one cycle, so ``_is_done()``
    would halt early and the assertion below would pass for the wrong reason.
    """
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ())]],
        worker=_failing_worker,
        tester=_passing_tester,
        max_cycles=2,
        fail_fast_cycles=50,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert o.cycles == 2
    assert len(rec.planner_calls) == 2
    assert o._tests_passed is True
    assert o._is_done() is False  # a failed task is unfinished work
    assert o._stop_reason() == "Reached max_cycles (2)."


async def test_an_exception_inside_a_cycle_is_contained(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crashing cycle is recorded, not propagated, and the loop still stops."""
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ())]],
        tester=_exploding_tester,
        max_cycles=1,
        fail_fast_cycles=50,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)  # returns normally

    assert o.cycles == 1
    assert o._hard_failure is True
    assert o._consecutive_non_progress == 1
    assert o._is_done() is False
    assert rec.worker_calls == ["1"]


async def test_fail_fast_cycles_bounds_a_hard_failure_loop(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_cycles=0`` (unlimited) is still bounded when every cycle raises."""
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ())]],
        tester=_exploding_tester,
        max_cycles=0,
        fail_fast_cycles=2,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert o.cycles == 2
    reason = o._stop_reason()
    assert reason is not None
    assert "2 consecutive cycles without progress" in reason
    assert "hard failure" in reason
    assert "fail_fast_cycles" in reason


async def test_non_progressing_cycles_are_bounded(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A persistently red suite must stop the loop instead of burning credits.

    With ``max_cycles=0`` (documented as unlimited) the only thing that can end
    this run is the non-progress budget: every cycle's tasks complete cleanly, so
    the *only* non-progress signal is the failing test run.
    """
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ())]],
        tester=_failing_tester,
        max_cycles=0,
        fail_fast_cycles=2,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert o.cycles == 2
    assert o._tests_passed is False
    assert all(task.status == TaskStatus.COMPLETED for task in o._get_tasks())
    assert o._is_done() is False
    reason = o._stop_reason()
    assert reason is not None
    assert "2 consecutive cycles without progress" in reason
    # The halt names the actual cause and the failing test run it came from.
    assert "failing test run" in reason
    assert "tests/test_widget.py::test_build" in (o.plan_context or "")


async def test_a_green_test_run_resets_the_non_progress_streak(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One productive cycle buys another ``fail_fast_cycles`` worth of retries.

    The design has to keep the green cycle from ending the run via ``_is_done()``.
    Each cycle plans a *fresh* ``Plan``, so a green cycle whose only task
    completed is genuinely finished work and the loop halts there -- the streak
    reset is never exercised. Cycle 2 therefore plans two tasks and the worker
    fails task ``"2"``, so the green test run leaves that task failed,
    ``_is_done()`` stays False, and the loop survives only because the streak was
    reset:

        cycle 1  task "1" ok      RED    -> streak 1
        cycle 2  "1" ok, "2" fail GREEN   -> streak 0 (reset, loop continues)
        cycle 3  task "1" ok      RED    -> streak 1
        cycle 4  task "1" ok      RED    -> streak 2 -> halt

    The streak only ever moves when ``_tests_passed is False``, so a green cycle
    cannot increment it in either variant -- it can only clear it. Remove the reset
    branch and cycle 2 leaves the streak at 1 instead of dropping it to 0, so the
    streak runs 1, 1, 2 across cycles 1, 2 and 3 and the run halts at cycle 3. The
    test therefore discriminates 4 against 3, not 4 against 2: ``o.cycles == 4``
    is the assertion that fails if the reset is taken out.
    """

    async def worker_that_fails_task_two(task: TaskModel, rec: _Recorder) -> WorkerResult:
        if task.id == "1":
            return await _ok_worker(task, rec)
        return await _failing_worker(task, rec)

    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ())], [("1", ()), ("2", ())], [("1", ())], [("1", ())]],
        worker=worker_that_fails_task_two,
        tester=_scripted_tester(RED, GREEN, RED, RED),
        max_cycles=0,
        fail_fast_cycles=2,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    # Cycles 3 and 4 are only reachable because cycle 2 reset the streak.
    assert o.cycles == 4
    assert len(rec.tester_calls) == 4
    assert rec.tester_calls[1][0] == o.goal
    # Cycle 2 was green but unfinished: that is what makes the reset observable.
    assert o._consecutive_non_progress == 2
    # The final halt is the non-progress budget, not a failed task in the last cycle.
    assert [task.status for task in o._get_tasks()] == [TaskStatus.COMPLETED]
    reason = o._stop_reason()
    assert reason is not None
    assert "2 consecutive cycles without progress" in reason
    assert "failing test run" in reason


async def test_empty_plan_does_not_count_as_non_progressing(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cycle that planned nothing is complete: no tests, no worker, no stall.

    Counting it as a failure would let a planner that keeps returning nothing
    spin the loop to the non-progress halt under ``max_cycles=0``.
    """
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[]],
        tester=_failing_tester,
        max_cycles=0,
        fail_fast_cycles=2,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert o.cycles == 1
    assert o._tests_passed is None
    assert o._consecutive_non_progress == 0
    assert o._hard_failure is False
    assert o._is_done() is True
    assert o._stop_reason() is None
    assert rec.tester_calls == []
    assert rec.worker_calls == []


def test_fail_fast_cycles_is_clamped_to_at_least_one(tmp_workspace: Settings) -> None:
    """``fail_fast_cycles=0`` must not switch the safety net off."""
    o = Orchestrator(
        "g", client=object(), settings=make_settings(workspace=tmp_workspace.workspace)
    )
    o.settings = make_settings(workspace=tmp_workspace.workspace, fail_fast_cycles=0)
    assert o._fail_fast_cycles() == 1
    o.settings = make_settings(workspace=tmp_workspace.workspace, fail_fast_cycles=-5)
    assert o._fail_fast_cycles() == 1


def test_fail_fast_cycles_falls_back_to_the_module_constant(tmp_workspace: Settings) -> None:
    """A settings object predating the field still gets a working bound."""
    o = Orchestrator(
        "g", client=object(), settings=make_settings(workspace=tmp_workspace.workspace)
    )
    o.settings = SimpleNamespace()  # type: ignore[assignment]
    assert o._fail_fast_cycles() == MAX_CONSECUTIVE_CYCLE_FAILURES


# --------------------------------------------------------------------------
# Concurrency and dependency scheduling
# --------------------------------------------------------------------------


async def test_max_parallel_tasks_bounds_worker_overlap(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Four independent tasks, two at a time: the peak is exactly 2.

    Equality, not ``<=``: a peak of 1 would mean the waves never overlapped, so
    the assertion has to rule out serialisation as well as unbounded fan-out.
    """
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ()), ("2", ()), ("3", ()), ("4", ())]],
        worker=_yielding_worker,
        max_parallel_tasks=2,
        max_cycles=1,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert sorted(rec.worker_calls) == ["1", "2", "3", "4"]
    assert rec.peak == 2
    assert rec.active == 0
    assert len(rec.entered) == len(rec.exited) == 4


async def test_a_dependency_is_finished_before_its_dependent_starts(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("A", ()), ("B", ("A",)), ("C", ("B",))]],
        worker=_yielding_worker,
        max_cycles=1,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert rec.exited_at("A") <= rec.entered_at("B")
    assert rec.exited_at("B") <= rec.entered_at("C")
    assert all(task.status == TaskStatus.COMPLETED for task in o._get_tasks())


async def test_an_unknown_dependency_id_counts_as_satisfied(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dependency the plan never mentioned must not strand the task."""
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("1", ("ghost",))]],
        max_cycles=1,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert rec.worker_calls == ["1"]
    assert [task.status for task in o._get_tasks()] == [TaskStatus.COMPLETED]


async def test_a_dependency_cycle_terminates_instead_of_deadlocking(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec, o = _build(
        tmp_workspace,
        monkeypatch,
        specs=[[("A", ("B",)), ("B", ("A",))]],
        max_cycles=1,
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert sorted(rec.worker_calls) == ["A", "B"]
    assert o.cycles == 1
    assert all(task.status == TaskStatus.COMPLETED for task in o._get_tasks())


# --------------------------------------------------------------------------
# _take_wave / _failure_context -- pure units
# --------------------------------------------------------------------------


def _wave_orchestrator(tmp_workspace: Settings) -> Orchestrator:
    return Orchestrator(
        "g", client=object(), settings=make_settings(workspace=tmp_workspace.workspace)
    )


def test_take_wave_splits_runnable_from_waiting(tmp_workspace: Settings) -> None:
    o = _wave_orchestrator(tmp_workspace)
    tasks = _plan([("A", ()), ("B", ()), ("C", ("A", "B"))]).tasks
    known = {task.id for task in tasks}

    runnable, waiting = o._take_wave(list(tasks), known, set())

    assert [task.id for task in runnable] == ["A", "B"]
    assert [task.id for task in waiting] == ["C"]


def test_take_wave_ignores_dependency_ids_outside_the_plan(tmp_workspace: Settings) -> None:
    o = _wave_orchestrator(tmp_workspace)
    tasks = _plan([("A", ("ghost",))]).tasks
    known = {task.id for task in tasks}

    runnable, waiting = o._take_wave(list(tasks), known, set())

    assert [task.id for task in runnable] == ["A"]
    assert waiting == []


def test_take_wave_skips_a_task_whose_dependency_can_never_complete(
    tmp_workspace: Settings,
) -> None:
    """The dependency left the pending set without completing, so it is dead.

    ``"A"`` has to be in ``known`` -- it is the *dead* dependency this is about.
    With it only in ``dependencies`` it would be filtered out as an unknown id
    (see ``test_take_wave_ignores_dependency_ids_outside_the_plan``), the
    dependency set would be empty, and ``B`` would be reported runnable.
    """
    o = _wave_orchestrator(tmp_workspace)
    tasks = _plan([("B", ("A",))]).tasks
    known = {"A", "B"}

    runnable, waiting = o._take_wave(list(tasks), known, set())

    assert runnable == []
    assert waiting == []
    # `==`, not `is`: `_take_wave` assigns the plain string "skipped" and
    # `TaskStatus` is a `str` enum, so identity comparison fails.
    assert tasks[0].status == TaskStatus.SKIPPED
    assert "did not complete" in (tasks[0].result or "")


def test_take_wave_reports_a_pure_cycle_as_all_waiting(tmp_workspace: Settings) -> None:
    """Nothing is runnable and nothing is dead: the caller has to break the tie."""
    o = _wave_orchestrator(tmp_workspace)
    tasks = _plan([("A", ("B",)), ("B", ("A",))]).tasks
    known = {task.id for task in tasks}

    runnable, waiting = o._take_wave(list(tasks), known, set())

    assert runnable == []
    assert [task.id for task in waiting] == ["A", "B"]
    assert all(task.status == TaskStatus.PENDING for task in tasks)


def test_failure_context_lists_every_failure(tmp_workspace: Settings) -> None:
    o = _wave_orchestrator(tmp_workspace)
    context = o._failure_context(["tests/test_a.py::test_x", "tests/test_b.py::test_y"])

    assert "failed its tests" in context
    assert "- tests/test_a.py::test_x" in context
    assert "- tests/test_b.py::test_y" in context
    assert context.index("test_a.py") < context.index("test_b.py")


def test_failure_context_without_failures_still_asks_for_a_recheck(
    tmp_workspace: Settings,
) -> None:
    o = _wave_orchestrator(tmp_workspace)
    context = o._failure_context([])

    assert context
    assert "Re-check the goal" in context


# --------------------------------------------------------------------------
# Planner shape probing
# --------------------------------------------------------------------------


def test_the_real_planner_satisfies_the_context_probe(tmp_workspace: Settings) -> None:
    """If this ever flips, failures stop reaching the planner and go to the goal."""
    o = _wave_orchestrator(tmp_workspace)
    o.planner = PlannerAgent(client=object(), settings=o.settings)
    assert o._planner_accepts_context() is True


def test_a_single_argument_planner_does_not_satisfy_the_context_probe(
    tmp_workspace: Settings,
) -> None:
    o = _wave_orchestrator(tmp_workspace)
    o.planner = _InlinePlanner(_Recorder())()
    assert o._planner_accepts_context() is False


def test_the_context_probe_is_memoised(tmp_workspace: Settings) -> None:
    """A cached ``False``/``True`` must not be recomputed per cycle."""
    o = _wave_orchestrator(tmp_workspace)
    o.planner = _InlinePlanner(_Recorder())()
    assert o._planner_accepts_context() is False
    o.planner = _Planner(_Recorder())()
    assert o._planner_accepts_context() is False  # cached, not re-probed
    assert o._planner_takes_context is False


# --------------------------------------------------------------------------
# Defensive coercion of a worker's return value
# --------------------------------------------------------------------------


async def test_a_plain_string_worker_result_is_coerced(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A worker that still returns a bare summary must not crash the wave."""

    async def string_worker(task: TaskModel, rec: _Recorder) -> str:
        return f"{task.id} wrote the module"

    rec, o = _build(
        tmp_workspace, monkeypatch, specs=[[("1", ())]], worker=string_worker, max_cycles=1
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    task = o._get_tasks()[0]
    assert task.status == TaskStatus.COMPLETED
    assert task.result == "1 wrote the module"
    assert o.cycles == 1
    assert o._is_done() is True


async def test_an_empty_string_worker_result_counts_as_a_failure(
    tmp_workspace: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``bool("")`` is False, so an empty summary is not reported as a success."""

    async def empty_worker(task: TaskModel, rec: _Recorder) -> str:
        return ""

    rec, o = _build(
        tmp_workspace, monkeypatch, specs=[[("1", ())]], worker=empty_worker, max_cycles=1
    )
    await asyncio.wait_for(o.run(), timeout=RUN_TIMEOUT)

    assert [task.status for task in o._get_tasks()] == [TaskStatus.FAILED]
    assert o._is_done() is False
