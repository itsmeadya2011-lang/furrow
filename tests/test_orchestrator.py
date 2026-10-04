"""Regression tests for the ``furrow.core.orchestrator`` rewrite.

Self-contained on purpose: this module defines its own stub client and fake
planner/worker/tester agents, so it needs no shared ``conftest.py``, no network,
no real LLM and no subprocess.

Two habits keep these tests honest:

* **No wall-clock races.** Concurrency is observed through an ``asyncio.Event``
  gate (:class:`_Gate`), never through ``sleep(0)`` yields: a parked worker
  cannot finish until the test opens the gate, so "how many tasks ran at once"
  and "in which order did they start" are facts about the orchestrator's
  scheduling decisions rather than about event-loop luck.
* **No settings leakage.** Every loop limit is installed with
  ``monkeypatch.setattr`` -- either on the shared ``furrow.config.settings``
  object or by rebinding ``orchestrator.settings`` -- so nothing survives the
  test that installed it.
"""

from __future__ import annotations

import asyncio
import io
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from furrow import config
from furrow.config import Plan, Settings, TaskModel
from furrow.config import TestResult as Verdict
from furrow.core import orchestrator as orch
from furrow.core.orchestrator import Orchestrator

GOAL = "ship the thing"

#: Deadlock/spin guard. Every run under test halts on its own terms, so blowing
#: through this budget *is* the failure mode -- it is never an assumption about
#: how long a healthy run is allowed to take.
RUN_TIMEOUT = 5.0

#: The shared ``furrow.config.settings`` singleton as it looked when this module
#: was imported, i.e. before any test had a chance to touch it. Snapshotting
#: beats hard-coding the field defaults so the leak guard below stays true even
#: when ``FURROW_*`` environment variables are set.
SETTINGS_AT_IMPORT: dict[str, Any] = {
    name: getattr(config.settings, name) for name in ("max_cycles", "max_parallel_tasks")
}


# --------------------------------------------------------------------------- #
# stubs
# --------------------------------------------------------------------------- #


@dataclass
class _StubClient:
    """Enough of ``LLMClient`` for the orchestrator to hold; never called."""

    settings: Any = field(
        default_factory=lambda: Settings(
            planner_model="stub-planner",
            worker_model="stub-worker",
            tester_model="stub-tester",
            workspace=Path.cwd(),
        )
    )


@dataclass
class _Recorder:
    """What the fake planner/tester saw, plus whatever a hook chooses to log."""

    planner_calls: int = 0
    planner_goals: list[str] = field(default_factory=list)
    tester_calls: int = 0
    tester_goals: list[str] = field(default_factory=list)
    worker_calls: list[str] = field(default_factory=list)
    events: list[str] = field(default_factory=list)


WorkerHook = Callable[[TaskModel], Awaitable[str]]


def _task(tid: str, deps: Sequence[str] = ()) -> TaskModel:
    return TaskModel(id=tid, description=f"work on {tid}", dependencies=list(deps))


def _plan(tasks: Sequence[TaskModel], rationale: str = "plan") -> Plan:
    return Plan(tasks=list(tasks), rationale=rationale)


def _passing(summary: str = "all green") -> Verdict:
    return Verdict(passed=True, summary=summary)


def _failing(*failures: str) -> Verdict:
    return Verdict(passed=False, summary="tests failed", failures=list(failures))


def _sequential_worker(rec: _Recorder) -> WorkerHook:
    """A worker hook that never awaits: it logs and returns immediately.

    Because it yields to nothing, the recorded start/done order is decided
    purely by the order in which the orchestrator hands coroutines to
    ``asyncio.gather``.  Use this when only "did every task run" matters, and
    :class:`_Gate` when overlap itself is under test.
    """

    async def hook(task: TaskModel) -> str:
        rec.worker_calls.append(task.id)
        rec.events.append(f"start:{task.id}")
        rec.events.append(f"done:{task.id}")
        return f"did {task.id}"

    return hook


def _raising_worker(rec: _Recorder, message: str = "worker exploded") -> WorkerHook:
    """A worker hook that always raises, so every task ends up ``failed``."""

    async def hook(task: TaskModel) -> str:
        rec.worker_calls.append(task.id)
        rec.events.append(f"start:{task.id}")
        rec.events.append(f"fail:{task.id}")
        raise RuntimeError(f"{message} on {task.id}")

    return hook


class _Gate:
    """A worker hook that parks every worker until the test opens the gate.

    Python 3.10 has no ``asyncio.Barrier`` (that is 3.11+), so the barrier is
    built from an ``asyncio.Condition`` that arriving workers notify on, plus an
    ``asyncio.Event`` that the test sets to let everyone through.  While the
    gate is shut the orchestrator cannot get past the current batch, which turns
    "how many tasks were in flight" and "what started before what" into
    deterministic observations.
    """

    def __init__(self) -> None:
        self.release = asyncio.Event()
        self.calls: list[str] = []
        self.events: list[str] = []
        self.started: list[str] = []
        self.finished: list[str] = []
        self.in_flight = 0
        self.peak = 0
        self._arrived = asyncio.Condition()

    async def hook(self, task: TaskModel) -> str:
        async with self._arrived:
            self.calls.append(task.id)
            self.started.append(task.id)
            self.events.append(f"start:{task.id}")
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self._arrived.notify_all()
        # Parked until the test says go. ``in_flight`` cannot drop before this
        # returns, so ``peak`` is a real high-water mark, not a sampling artefact.
        await self.release.wait()
        async with self._arrived:
            self.in_flight -= 1
            self.finished.append(task.id)
            self.events.append(f"done:{task.id}")
            self._arrived.notify_all()
        return f"did {task.id}"

    async def wait_for_started(self, count: int) -> None:
        """Block until ``count`` workers are parked at the gate.

        ``asyncio.Condition.wait_for`` evaluates the predicate first and only
        then delegates to ``Condition.wait()``, which *requires* the caller to
        already hold the lock -- it raises ``RuntimeError: cannot wait on
        un-acquired lock`` otherwise.  So the lock is taken here; without it
        this helper blows up the moment the gate is genuinely shut, which is
        exactly the state it exists to observe.

        ``asyncio.wait_for`` is a deadlock guard, nothing more: if the
        orchestrator fails to schedule ``count`` workers the test fails with a
        ``TimeoutError`` instead of hanging forever.
        """

        async def _parked() -> None:
            async with self._arrived:
                await self._arrived.wait_for(lambda: len(self.started) >= count)

        await asyncio.wait_for(_parked(), RUN_TIMEOUT)


def _rig(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plans: Sequence[Plan],
    rec: _Recorder | None = None,
    tester_results: Sequence[Verdict] = (),
    worker_hook: WorkerHook | None = None,
) -> _Recorder:
    """Install fake agents into the orchestrator module; return the recorder.

    ``rec`` is the single object every fake writes to, so a caller that builds
    its own worker hook must hand that same recorder in -- otherwise the counts
    it asserts on belong to a throwaway.

    The fake planner replays ``plans`` and the fake tester replays
    ``tester_results`` (the last entry repeats), so a loop that runs away shows
    up as a wrong call count rather than as an exception.
    """
    if rec is None:
        rec = _Recorder()
    plan_queue = list(plans)
    tester_queue = list(tester_results)
    hook = _sequential_worker(rec) if worker_hook is None else worker_hook

    # The module builds one module-level Console and passes it explicitly to
    # rich's Status spinners, so redirecting that object silences the panel
    # output *and* the live spinners.  Nothing here asserts on console text.
    monkeypatch.setattr(orch, "console", Console(file=io.StringIO(), width=200))

    class FakePlanner:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.client = client

        async def plan(self, goal: str) -> Plan:
            rec.planner_calls += 1
            rec.planner_goals.append(goal)
            return plan_queue[min(rec.planner_calls - 1, len(plan_queue) - 1)]

    class FakeWorker:
        def __init__(self, task: TaskModel, client: Any = None, settings: Any = None) -> None:
            self.task = task

        async def run(self) -> str:
            return await hook(self.task)

    class FakeTester:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            self.client = client

        async def run(self, goal: str, tasks: list[TaskModel]) -> Verdict:
            rec.tester_calls += 1
            rec.tester_goals.append(goal)
            if not tester_queue:
                return _passing()
            return tester_queue[min(rec.tester_calls - 1, len(tester_queue) - 1)]

    monkeypatch.setattr(orch, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orch, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orch, "TesterAgent", FakeTester)
    return rec


def _budgeted(monkeypatch: pytest.MonkeyPatch, **overrides: Any) -> None:
    """Set loop limits on the live ``furrow.config.settings`` singleton.

    ``monkeypatch.setattr`` on the instance records the original value, so the
    real singleton is restored at teardown and cannot leak into another test.
    """
    for name, value in overrides.items():
        monkeypatch.setattr(config.settings, name, value)


async def _finish(run: asyncio.Task[None]) -> None:
    """Await an in-flight ``Orchestrator.run()`` under the spin guard."""
    await asyncio.wait_for(run, RUN_TIMEOUT)


# --------------------------------------------------------------------------- #
# an empty plan halts after exactly one cycle
# --------------------------------------------------------------------------- #


async def test_empty_plan_halts_after_one_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    _budgeted(monkeypatch, max_cycles=0)
    rec = _rig(monkeypatch, plans=[_plan([], rationale="already done")])

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert rec.planner_calls == 1, "the planner must be asked exactly once"
    assert target.cycles == 1
    assert target._no_tasks_planned is True
    assert rec.tester_calls == 0, "nothing to execute, so nothing to test"
    # The run stopped because the planner had nothing left to say, *not* because
    # an empty task list masqueraded as a finished goal.
    assert target._is_done() is False
    assert target._get_tasks() == []


# --------------------------------------------------------------------------- #
# max_cycles stops the loop at exactly N cycles
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("budget", [1, 2, 3], ids=["max_cycles=1", "max_cycles=2", "max_cycles=3"])
async def test_max_cycles_stops_loop_at_exactly_n_cycles(
    monkeypatch: pytest.MonkeyPatch, budget: int
) -> None:
    # Lifting the failure cap leaves the cycle budget as the only terminator,
    # and a worker that always raises keeps the goal from ever being "done".
    monkeypatch.setattr(orch, "MAX_CONSECUTIVE_FAILURES", 99)
    _budgeted(monkeypatch, max_cycles=budget)
    rec = _Recorder()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1")], rationale="always the same work")],
        tester_results=[_failing("tests/test_a.py::test_x FAILED")],
        worker_hook=_raising_worker(rec),
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.cycles == budget
    assert rec.planner_calls == budget
    assert rec.tester_calls == budget
    # Stopped on the budget, not on the failure cap that was lifted to 99.
    assert rec.planner_calls < orch.MAX_CONSECUTIVE_FAILURES


# --------------------------------------------------------------------------- #
# max_cycles = 0 is uncapped
# --------------------------------------------------------------------------- #


async def test_max_cycles_zero_is_uncapped_by_the_cycle_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero budget must not stop the loop; the empty plan has to."""
    _budgeted(monkeypatch, max_cycles=0)
    rec = _Recorder()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[
            _plan([_task("1")], rationale="fails"),
            _plan([], rationale="nothing left to do"),
        ],
        tester_results=[_failing("still red")],
        worker_hook=_raising_worker(rec),
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    # Two cycles ran, so the zero budget was not what stopped the loop: cycle 1
    # left a failed task (hence not "done") and cycle 2 produced an empty plan.
    assert orch._settings().max_cycles == 0
    assert target.cycles == 2
    assert rec.planner_calls == 2
    assert rec.tester_calls == 1, "the empty plan short-circuits before testing"
    assert target._no_tasks_planned is True
    # The failure context from cycle 1 was handed to the cycle 2 planner.
    assert "still red" in rec.planner_goals[1]


async def test_max_cycles_zero_still_halts_on_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same claim as above, but terminated by the failure counter instead.

    Split out from the empty-plan case so both uncapped terminators are pinned
    down independently.
    """
    _budgeted(monkeypatch, max_cycles=0)
    rec = _Recorder()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1")], rationale="try again")],
        tester_results=[_failing("boom")],
        worker_hook=_raising_worker(rec),
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.cycles == orch.MAX_CONSECUTIVE_FAILURES
    assert target._consecutive_failures == orch.MAX_CONSECUTIVE_FAILURES
    assert rec.planner_calls == orch.MAX_CONSECUTIVE_FAILURES


# --------------------------------------------------------------------------- #
# goal is write-once; failure context is layered onto the planning goal
# --------------------------------------------------------------------------- #


async def test_failure_context_reaches_planner_without_clobbering_goal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _budgeted(monkeypatch, max_cycles=2)
    rec = _Recorder()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1")], rationale="fix it")],
        tester_results=[_failing("tests/test_a.py::test_x FAILED", "tests/test_b.py::test_y")],
        worker_hook=_raising_worker(rec, message="compile error"),
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.goal == GOAL, "self.goal must be write-once"
    assert target._failure_context is not None
    assert "tests/test_a.py::test_x FAILED" in target._failure_context
    assert "tests/test_b.py::test_y" in target._failure_context

    # First cycle gets the bare goal; the retry is told what went wrong.
    assert rec.planner_goals[0] == GOAL
    assert rec.planner_goals[1].startswith(GOAL)
    assert target._failure_context in rec.planner_goals[1]
    assert "tests/test_a.py::test_x FAILED" in rec.planner_goals[1]
    # ...while the tester always gets the pristine goal, never the annotated one.
    assert rec.tester_goals == [GOAL, GOAL]


async def test_planning_goal_is_bare_goal_without_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _budgeted(monkeypatch, max_cycles=1)
    rec = _rig(monkeypatch, plans=[_plan([_task("1")])], tester_results=[_passing()])

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert rec.planner_goals == [GOAL]
    assert target._failure_context is None
    assert target._consecutive_failures == 0
    assert target.cycles == 1


# --------------------------------------------------------------------------- #
# repeated failures halt
# --------------------------------------------------------------------------- #


async def test_repeated_failures_halt_after_max_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The "fix the same failing tests forever" loop is bounded.

    ``_finish`` is the anti-hang witness: a regression that never reaches the
    failure cap shows up as a TimeoutError instead of a stuck test run.
    """
    _budgeted(monkeypatch, max_cycles=0)
    rec = _Recorder()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1")], rationale="try again")],
        tester_results=[_failing("boom")],
        worker_hook=_raising_worker(rec),
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.cycles == orch.MAX_CONSECUTIVE_FAILURES
    assert target._consecutive_failures == orch.MAX_CONSECUTIVE_FAILURES
    assert rec.planner_calls == orch.MAX_CONSECUTIVE_FAILURES
    assert rec.tester_calls == orch.MAX_CONSECUTIVE_FAILURES
    assert rec.worker_calls == ["1"] * orch.MAX_CONSECUTIVE_FAILURES


async def test_passing_cycle_resets_the_failure_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A green cycle with no failed task must clear the counter and the context.

    Two counted cycles are followed by one good one.  If the reset were missing,
    the counter would still read 3 (== ``MAX_CONSECUTIVE_FAILURES``) and the
    "Fix failing tests" context would still be attached to the goal, so both
    assertions below discriminate the reset from its absence.
    """
    _budgeted(monkeypatch, max_cycles=4)
    rec = _Recorder()

    async def hook(task: TaskModel) -> str:
        rec.worker_calls.append(task.id)
        if task.id in {"1", "2"}:
            raise RuntimeError(f"cannot build {task.id}")
        return f"did {task.id}"

    _rig(
        monkeypatch,
        rec=rec,
        plans=[
            _plan([_task("1")], rationale="broken"),
            _plan([_task("2")], rationale="still broken"),
            _plan([_task("3")], rationale="fixed"),
        ],
        tester_results=[_failing("boom"), _failing("boom again"), _passing()],
        worker_hook=hook,
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.cycles == 3, "the 4-cycle budget must not be what stopped the run"
    assert rec.planner_calls == 3
    assert rec.tester_calls == 3
    assert target._consecutive_failures == 0, "a green cycle must reset the counter"
    assert target._failure_context is None, "a green cycle must drop the failure context"
    # The context was still being fed to the planner while it was relevant.
    assert "boom again" in rec.planner_goals[2]


# --------------------------------------------------------------------------- #
# max_parallel_tasks caps real concurrency
# --------------------------------------------------------------------------- #


async def test_max_parallel_tasks_caps_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    _budgeted(monkeypatch, max_cycles=1, max_parallel_tasks=2)
    rec = _Recorder()
    gate = _Gate()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1"), _task("2"), _task("3"), _task("4")])],
        tester_results=[_passing()],
        worker_hook=gate.hook,
    )

    target = Orchestrator(GOAL, client=_StubClient())
    run = asyncio.create_task(target.run())

    # The gate is shut, so the orchestrator is frozen on its first batch. If the
    # cap were ignored, all four tasks would be parked here instead of two.
    await gate.wait_for_started(2)
    assert gate.in_flight == 2
    assert sorted(gate.started) == ["1", "2"], "a third task started before the first two finished"

    gate.release.set()
    await _finish(run)

    assert gate.peak == 2, f"peak concurrency was {gate.peak}"
    assert sorted(gate.calls) == ["1", "2", "3", "4"], "every task must run exactly once"
    assert sorted(gate.finished) == ["1", "2", "3", "4"]
    assert all(t.status == "completed" for t in target._get_tasks())
    assert all(t.result is not None for t in target._get_tasks())
    assert rec.tester_calls == 1
    assert target.cycles == 1


async def test_max_parallel_tasks_zero_runs_everything_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``cap <= 0`` means uncapped: the whole plan is one batch."""
    _budgeted(monkeypatch, max_cycles=1, max_parallel_tasks=0)
    rec = _Recorder()
    gate = _Gate()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1"), _task("2"), _task("3")])],
        tester_results=[_passing()],
        worker_hook=gate.hook,
    )

    target = Orchestrator(GOAL, client=_StubClient())
    run = asyncio.create_task(target.run())

    await gate.wait_for_started(3)
    assert gate.in_flight == 3, "an uncapped run must schedule the whole plan at once"

    gate.release.set()
    await _finish(run)

    assert gate.peak == 3
    assert sorted(gate.calls) == ["1", "2", "3"]
    assert all(t.status == "completed" for t in target._get_tasks())


# --------------------------------------------------------------------------- #
# dependency ordering
# --------------------------------------------------------------------------- #


async def test_dependent_task_waits_for_its_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _budgeted(monkeypatch, max_cycles=1, max_parallel_tasks=5)
    # The dependent task is listed first, so only a real dependency check can
    # keep it from being scheduled alongside task "1".
    plan = _plan([_task("2", deps=["1"]), _task("1"), _task("3", deps=["2"])])
    rec = _Recorder()
    gate = _Gate()
    _rig(
        monkeypatch, rec=rec, plans=[plan], tester_results=[_passing()], worker_hook=gate.hook
    )

    target = Orchestrator(GOAL, client=_StubClient())
    run = asyncio.create_task(target.run())

    # While "1" is parked at the gate, its dependents must not have started.
    await gate.wait_for_started(1)
    assert gate.started == ["1"]

    gate.release.set()
    await _finish(run)

    # Strict alternation: because nothing could finish before the gate opened,
    # two adjacent "start" events would mean the tasks really did overlap.
    assert gate.events == [
        "start:1",
        "done:1",
        "start:2",
        "done:2",
        "start:3",
        "done:3",
    ]
    assert gate.peak == 1, "a dependency chain must not run tasks in parallel"
    assert sorted(gate.calls) == ["1", "2", "3"]
    assert all(t.status == "completed" for t in target._get_tasks())


async def test_cap_defers_extra_tasks_and_dependencies_are_still_honoured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tasks 1 and 2 fill the cap, so ready task 3 waits for a later batch."""
    _budgeted(monkeypatch, max_cycles=1, max_parallel_tasks=2)
    plan = _plan([_task("1"), _task("2"), _task("3", deps=["1"])])
    rec = _Recorder()
    gate = _Gate()
    _rig(
        monkeypatch, rec=rec, plans=[plan], tester_results=[_passing()], worker_hook=gate.hook
    )

    target = Orchestrator(GOAL, client=_StubClient())
    run = asyncio.create_task(target.run())

    await gate.wait_for_started(2)
    assert sorted(gate.started) == ["1", "2"], "ready task 3 should be deferred by the cap"
    assert gate.in_flight == 2

    gate.release.set()
    await _finish(run)

    assert gate.peak == 2
    assert gate.events == ["start:1", "start:2", "done:1", "done:2", "start:3", "done:3"]
    assert sorted(gate.calls) == ["1", "2", "3"]
    assert all(t.status == "completed" for t in target._get_tasks())


# --------------------------------------------------------------------------- #
# cyclic / missing dependencies
# --------------------------------------------------------------------------- #


async def test_cyclic_dependencies_run_instead_of_deadlocking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _budgeted(monkeypatch, max_cycles=1, max_parallel_tasks=2)
    plan = _plan([_task("A", deps=["B"]), _task("B", deps=["A"])])
    rec = _Recorder()
    gate = _Gate()
    _rig(
        monkeypatch, rec=rec, plans=[plan], tester_results=[_passing()], worker_hook=gate.hook
    )

    target = Orchestrator(GOAL, client=_StubClient())
    run = asyncio.create_task(target.run())

    # Nothing can ever become ready, so the orchestrator must relax the graph and
    # run both tasks; the gate is released immediately, only to prove the batch
    # is really in flight rather than a silent drop.
    await gate.wait_for_started(2)
    gate.release.set()
    await _finish(run)

    assert sorted(gate.calls) == ["A", "B"], "a cyclic graph must not silently drop a task"
    assert all(t.status == "completed" for t in target._get_tasks())
    assert all(t.result is not None for t in target._get_tasks())
    assert target.cycles == 1, "the cycle must terminate"


async def test_missing_dependency_still_runs_the_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _budgeted(monkeypatch, max_cycles=1, max_parallel_tasks=2)
    plan = _plan([_task("1", deps=["does-not-exist"]), _task("2")])
    rec = _Recorder()
    gate = _Gate()
    _rig(
        monkeypatch, rec=rec, plans=[plan], tester_results=[_passing()], worker_hook=gate.hook
    )

    target = Orchestrator(GOAL, client=_StubClient())
    run = asyncio.create_task(target.run())

    # Task "1" depends on a task that does not exist, so it is *not* ready and
    # only "2" is scheduled in the first batch.  Waiting for both arrivals before
    # opening the gate would deadlock on the orchestrator's own (correct) choice
    # to schedule one worker, so the gate is opened after the first arrival and
    # the second arrival is then awaited.
    await gate.wait_for_started(1)
    assert gate.started == ["2"], "the task with a missing dependency must not be first"

    gate.release.set()
    await gate.wait_for_started(2)
    await _finish(run)

    assert sorted(gate.calls) == ["1", "2"]
    assert gate.started == ["2", "1"], "the blocked task runs once the graph is relaxed"
    tasks = {t.id: t for t in target._get_tasks()}
    assert tasks["1"].status == "completed"
    assert tasks["1"].result is not None, "the task must not be dropped silently"
    assert target.cycles == 1


# --------------------------------------------------------------------------- #
# _is_done() must not report "done" for an empty task list
# --------------------------------------------------------------------------- #


def test_is_done_is_false_for_an_empty_task_list() -> None:
    """ROOT-CAUSE REGRESSION TEST -- read this one twice.

    The original stub ``_get_tasks()`` returned ``[]``, and an early
    ``_is_done()`` of the form ``return not any(t.status != "completed" for t in
    tasks)`` is vacuously ``True`` for an empty list.  That made the whole
    "infinite" loop print "Goal complete" and halt after a single cycle, no
    matter what the planner or the workers had done.

    An empty retained task list therefore means "nothing has been planned yet",
    which is emphatically *not* a finished goal, and ``_is_done()`` must say so
    before any plan exists, after a failed cycle, and mid-run.
    """
    target = Orchestrator(GOAL, client=_StubClient())

    assert target._get_tasks() == [], "no plan has been stored yet"
    assert target.cycles == 0
    assert target._is_done() is False, "an empty task list is not a finished goal"


def test_is_done_requires_every_retained_task_to_be_completed() -> None:
    target = Orchestrator(GOAL, client=_StubClient())
    target._store_plan(_plan([_task("1"), _task("2")]))

    assert target._is_done() is False
    target._get_tasks()[0].status = "completed"
    assert target._is_done() is False
    target._get_tasks()[1].status = "completed"
    assert target._is_done() is True
    target._get_tasks()[0].status = "failed"
    assert target._is_done() is False, "a failed task is never 'done'"
    target._get_tasks()[0].status = "pending"
    assert target._is_done() is False


def test_is_done_requires_a_green_test_verdict() -> None:
    """Same accessor, second root cause: task statuses alone are not a verdict.

    ``_cycle()`` records the tester's verdict in ``_tests_passed`` (``None``
    until a cycle has run) and ``_is_done()`` must refuse to call the goal
    finished while that verdict is red.  This is the unit-level twin of
    ``test_failing_tests_after_completed_tasks_still_get_a_repair_cycle``,
    which reaches the same line through a real run.
    """
    target = Orchestrator(GOAL, client=_StubClient())
    target._store_plan(_plan([_task("1")]))
    target._get_tasks()[0].status = "completed"

    assert target._tests_passed is None, "no cycle has run yet"
    assert target._is_done() is True, "no red verdict on record is not a failure"

    target._tests_passed = False
    assert target._is_done() is False, "completed tasks with red tests are not 'done'"

    target._tests_passed = True
    assert target._is_done() is True


# --------------------------------------------------------------------------- #
# a raising worker is recorded, not fatal
# --------------------------------------------------------------------------- #


async def test_worker_exception_is_recorded_as_a_failed_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _budgeted(monkeypatch, max_cycles=1, max_parallel_tasks=2)
    rec = _Recorder()

    async def hook(task: TaskModel) -> str:
        rec.worker_calls.append(task.id)
        rec.events.append(f"start:{task.id}")
        if task.id == "bad":
            rec.events.append("fail:bad")
            raise RuntimeError("worker exploded on bad")
        rec.events.append(f"done:{task.id}")
        return f"did {task.id}"

    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("bad"), _task("good")])],
        tester_results=[_failing("good task still ran")],
        worker_hook=hook,
    )

    target = Orchestrator(GOAL, client=_StubClient())
    # Must not raise: asyncio.gather(return_exceptions=True) traps the failure.
    await _finish(asyncio.create_task(target.run()))

    tasks = {t.id: t for t in target._get_tasks()}
    assert sorted(rec.worker_calls) == ["bad", "good"]
    assert tasks["bad"].status == "failed"
    assert "worker exploded on bad" in (tasks["bad"].result or "")
    assert tasks["good"].status == "completed"
    assert tasks["good"].result == "did good"
    # The cycle continued past the failing worker: the tests still ran.
    assert rec.tester_calls == 1
    assert rec.tester_goals == [GOAL]


async def test_all_tasks_failing_still_terminates_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _budgeted(monkeypatch, max_cycles=0)
    rec = _Recorder()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1"), _task("2")])],
        tester_results=[_failing("nope")],
        worker_hook=_raising_worker(rec),
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.cycles == orch.MAX_CONSECUTIVE_FAILURES
    assert rec.worker_calls == ["1", "2"] * orch.MAX_CONSECUTIVE_FAILURES
    assert all(t.status == "failed" for t in target._get_tasks())
    # Failing tests own the context here; "Repair or redo" belongs to the
    # green-tests-but-a-task-failed branch, which is a different test.
    assert (target._failure_context or "").startswith("Fix failing tests:")
    assert "nope" in (target._failure_context or "")


async def test_passed_tests_with_a_failed_task_sets_the_repair_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of ``_track_progress``: tests green, a task still red."""
    _budgeted(monkeypatch, max_cycles=2, max_parallel_tasks=2)
    rec = _Recorder()

    async def hook(task: TaskModel) -> str:
        rec.worker_calls.append(task.id)
        if task.id == "bad":
            raise RuntimeError("worker exploded on bad")
        return f"did {task.id}"

    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("bad"), _task("good")])],
        tester_results=[_passing()],
        worker_hook=hook,
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target._failure_context == "Repair or redo the following failed tasks:\nbad"
    assert "Repair or redo" in rec.planner_goals[1], "the next planner call must see the repair list"
    assert "bad" in rec.planner_goals[1]
    assert target.goal == GOAL, "the annotated goal never overwrites self.goal"
    assert target._consecutive_failures == 2
    # "good" was already completed, so only the failed task is retried.
    assert rec.worker_calls == ["bad", "good", "bad"]
    assert rec.tester_calls == 2
    assert rec.tester_goals == [GOAL, GOAL]


async def test_failing_tests_after_completed_tasks_still_get_a_repair_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed tasks are not verified work.

    This test was an ``xfail(strict=True)`` tripwire for a real orchestrator
    bug: ``_is_done()`` decided completion from task statuses alone, so a cycle
    in which every task *completed* but the tests *failed* printed "Goal
    complete. Halting." and broke out of ``run()`` before the failure counter
    was ever consulted -- even though the same cycle had printed "Will attempt
    fix in next cycle."  The bug is fixed (``_cycle()`` records the test
    verdict in ``self._tests_passed`` and ``_is_done()`` refuses a red
    verdict), so the tripwire has served its purpose and this is now an
    ordinary test: it must pass.
    """
    _budgeted(monkeypatch, max_cycles=2)
    rec = _rig(
        monkeypatch,
        plans=[_plan([_task("1")]), _plan([_task("2")], rationale="repair")],
        tester_results=[_failing("boom"), _passing()],
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert rec.planner_calls == 2, "the failing tests deserve a second, repairing cycle"
    assert rec.tester_calls == 2
    assert target.cycles == 2
    # The first cycle really did look complete from the task statuses alone --
    # that is what used to be enough to halt the run.
    assert target._consecutive_failures == 0, "the repair cycle went green"
    assert "boom" in rec.planner_goals[1], "the second planner call sees the red tests"
    assert rec.planner_goals[1].startswith(GOAL), "the annotated goal never becomes self.goal"
    assert target.goal == GOAL


async def test_completed_tasks_with_green_tests_finish_promptly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Converse of the repair-cycle test: a verified cycle is not looped on.

    The cycle budget is uncapped here, so ``_is_done()`` is the only thing that
    can stop the run.  One cycle and one planner call is the proof that a fully
    successful cycle is recognised immediately -- the repair-cycle test above
    only proves the opposite direction, and a regression that made every green
    cycle run ``max_cycles`` times would pass that test.
    """
    _budgeted(monkeypatch, max_cycles=0, max_parallel_tasks=2)
    rec = _rig(
        monkeypatch,
        plans=[
            _plan([_task("1"), _task("2")]),
            _plan([_task("3")], rationale="must never be planned"),
        ],
        tester_results=[_passing()],
    )

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.cycles == 1, "a green cycle with every task completed halts at once"
    assert rec.planner_calls == 1
    assert rec.tester_calls == 1
    assert rec.worker_calls == ["1", "2"]
    assert all(t.status == "completed" for t in target._get_tasks())
    assert target._tests_passed is True
    assert target._is_done() is True
    assert target._consecutive_failures == 0
    assert target._failure_context is None


# --------------------------------------------------------------------------- #
# both settings-override styles used by _settings()
# --------------------------------------------------------------------------- #


async def test_cycle_budget_can_be_set_on_the_config_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _budgeted(monkeypatch, max_cycles=2)
    monkeypatch.setattr(orch, "MAX_CONSECUTIVE_FAILURES", 99)
    rec = _Recorder()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1")])],
        tester_results=[_failing("boom")],
        worker_hook=_raising_worker(rec),
    )

    assert orch._settings() is config.settings
    assert orch._settings().max_cycles == 2

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.cycles == 2
    assert rec.planner_calls == 2


async def test_cycle_budget_can_be_set_by_rebinding_the_module_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    override = Settings(max_cycles=3, max_parallel_tasks=2)
    monkeypatch.setattr(orch, "settings", override)
    monkeypatch.setattr(orch, "MAX_CONSECUTIVE_FAILURES", 99)
    rec = _Recorder()
    _rig(
        monkeypatch,
        rec=rec,
        plans=[_plan([_task("1"), _task("2")])],
        tester_results=[_failing("boom")],
        worker_hook=_raising_worker(rec),
    )

    # The rebind is picked up through the identity check in _settings().
    assert orch._settings() is override
    assert orch._settings().max_cycles == 3
    assert orch._settings().max_parallel_tasks == 2

    target = Orchestrator(GOAL, client=_StubClient())
    await _finish(asyncio.create_task(target.run()))

    assert target.cycles == 3
    assert rec.planner_calls == 3


def test_shared_settings_singleton_was_not_mutated() -> None:
    """Guards against a test leaking its loop limits into the shared singleton.

    Comparing against the import-time snapshot rather than the field defaults
    keeps this true even when ``FURROW_MAX_CYCLES`` and friends are exported.
    """
    assert {
        name: getattr(config.settings, name) for name in SETTINGS_AT_IMPORT
    } == SETTINGS_AT_IMPORT
    assert orch.settings is config.settings
