"""Tests for the orchestrator's loop logic and state persistence.

These tests avoid real LLM calls by stubbing the planner/worker/tester agents.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


class _StubPlanner:
    """Planner that returns a fixed list of plans in order."""

    def __init__(self, plans: list[Plan]) -> None:
        self._plans = list(plans)
        self.calls = 0

    async def plan(self, goal: str, max_attempts: int = 3) -> Plan:
        self.calls += 1
        return self._plans.pop(0)


class _StubWorker:
    """Worker that returns 'done' immediately."""

    def __init__(self, task: TaskModel, client: Any = None) -> None:
        self.task = task

    async def run(self) -> str:
        return "done"


class _StubTester:
    def __init__(self, client: Any = None, settings: Any = None) -> None:
        pass

    async def run(self, goal: str, tasks: list[TaskModel]) -> Any:
        from furrow.config import TestResult

        return TestResult(passed=True, summary="ok", failures=[])


class _FailingTester:
    """Tester that always reports failure -> goal not done, loop continues."""

    def __init__(self, client: Any = None, settings: Any = None) -> None:
        pass

    async def run(self, goal: str, tasks: list[TaskModel]) -> Any:
        from furrow.config import TestResult

        return TestResult(passed=False, summary="nope", failures=["x == y"])


class _FailingWorker:
    """Worker that always raises -> task marked 'failed'."""

    def __init__(self, task: TaskModel, client: Any = None) -> None:
        self.task = task

    async def run(self) -> str:
        raise RuntimeError("simulated worker failure")


class _TrackingWorker:
    """Worker that tracks how many run concurrently."""

    _current = 0
    _max = 0

    def __init__(self, task: TaskModel, client: Any = None) -> None:
        self.task = task

    async def run(self) -> str:
        import asyncio

        _TrackingWorker._current += 1
        _TrackingWorker._max = max(_TrackingWorker._max, _TrackingWorker._current)
        await asyncio.sleep(0.05)
        _TrackingWorker._current -= 1
        return "done"


def _make_orch(max_cycles: int = 0, max_parallel_tasks: int = 5) -> Orchestrator:
    orch = Orchestrator(goal="do the thing")
    orch.settings = Settings(
        max_cycles=max_cycles, max_parallel_tasks=max_parallel_tasks
    )
    orch.client.settings = orch.settings
    orch.cycles = 0
    orch.tasks = []
    return orch


def test_is_done_true_when_no_tasks():
    orch = _make_orch()
    orch.tasks = []
    assert orch._is_done() is True


def test_is_done_true_when_all_completed():
    orch = _make_orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_false_when_any_failed():
    orch = _make_orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_false_when_partial():
    orch = _make_orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_loop_halts_after_completion(monkeypatch):
    """A fully-completed cycle must halt (regression: previously halted after 1
    cycle even with un-stored/failing tasks because _get_tasks() returned [])."""
    import furrow.core.orchestrator as orch_mod

    orch = _make_orch()
    monkeypatch.setattr(orch_mod, "WorkerAgent", _StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", _StubTester)
    orch.planner = _StubPlanner(
        plans=[Plan(tasks=[TaskModel(id="1", description="a")], rationale="x")]
    )

    await orch.run()
    assert orch.cycles == 1
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_on_cycle_callback_receives_summary(monkeypatch):
    """The optional on_cycle callback must receive a plain-text summary per cycle
    (used by the web UI to stream progress over the websocket)."""
    import furrow.core.orchestrator as orch_mod

    sent: list[str] = []

    async def cb(text: str) -> None:
        sent.append(text)

    orch = _make_orch()
    orch.on_cycle = cb
    monkeypatch.setattr(orch_mod, "WorkerAgent", _StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", _StubTester)
    orch.planner = _StubPlanner(
        plans=[Plan(tasks=[TaskModel(id="1", description="a")], rationale="x")]
    )

    await orch.run()
    assert len(sent) == 1
    assert "Cycle 1" in sent[0]
    assert "completed" in sent[0]


@pytest.mark.asyncio
async def test_loop_runs_multiple_cycles_then_completes(monkeypatch):
    """Cycle 1 fails (forcing a second cycle); cycle 2 succeeds and halts."""
    import furrow.core.orchestrator as orch_mod

    class _FailOnceWorker:
        _calls = 0

        def __init__(self, task: TaskModel, client: Any = None) -> None:
            self.task = task

        async def run(self) -> str:
            _FailOnceWorker._calls += 1
            if _FailOnceWorker._calls == 1:
                raise RuntimeError("boom")
            return "done"

    orch = _make_orch()
    monkeypatch.setattr(orch_mod, "WorkerAgent", _FailOnceWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", _StubTester)
    orch.planner = _StubPlanner(
        plans=[
            Plan(tasks=[TaskModel(id="1", description="a")], rationale="x"),
            Plan(tasks=[TaskModel(id="2", description="b")], rationale="y"),
        ]
    )

    await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_max_cycles_limits_loop(monkeypatch):
    """With a perpetually-failing task, the loop must stop at max_cycles."""
    import furrow.core.orchestrator as orch_mod

    orch = _make_orch(max_cycles=3)
    monkeypatch.setattr(orch_mod, "WorkerAgent", _FailingWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", _StubTester)
    orch.planner = _StubPlanner(
        plans=[
            Plan(tasks=[TaskModel(id="1", description="a")], rationale="x"),
            Plan(tasks=[TaskModel(id="2", description="a")], rationale="x"),
            Plan(tasks=[TaskModel(id="3", description="a")], rationale="x"),
        ]
    )

    await orch.run()
    assert orch.cycles == 3


@pytest.mark.asyncio
async def test_max_parallel_tasks_enforced(monkeypatch):
    """Workers must never run more than max_parallel_tasks concurrently."""
    _TrackingWorker._current = 0
    _TrackingWorker._max = 0
    import furrow.core.orchestrator as orch_mod

    orch = _make_orch(max_parallel_tasks=2)
    monkeypatch.setattr(orch_mod, "WorkerAgent", _TrackingWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", _StubTester)

    tasks = [TaskModel(id=str(i), description=f"task {i}") for i in range(5)]
    orch.planner = _StubPlanner(plans=[Plan(tasks=tasks, rationale="x")])

    await orch.run()
    assert _TrackingWorker._max == 2


@pytest.mark.asyncio
async def test_loop_continues_when_tests_fail(monkeypatch):
    """If tasks complete but tests fail, the loop must NOT halt (it should
    retry). Bounded by max_cycles to keep the test terminating."""
    import furrow.core.orchestrator as orch_mod

    orch = _make_orch(max_cycles=2)
    monkeypatch.setattr(orch_mod, "WorkerAgent", _StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", _FailingTester)
    orch.planner = _StubPlanner(
        plans=[
            Plan(tasks=[TaskModel(id="1", description="a")], rationale="x"),
            Plan(tasks=[TaskModel(id="2", description="b")], rationale="y"),
        ]
    )

    await orch.run()
    # Tests keep failing, so _is_done() is always False -> only max_cycles stops it
    assert orch.cycles == 2


def test_state_persistence_roundtrip(tmp_path: Path):
    orch = _make_orch()
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    orch.current_plan = Plan(tasks=orch.tasks, rationale="x")
    orch.cycles = 2

    state_file = tmp_path / "state.json"
    orch.save_state(state_file)

    loaded = Orchestrator.load_state(state_file)
    assert loaded.goal == orch.goal
    assert loaded.cycles == 2
    assert loaded.current_plan is not None
    assert len(loaded.tasks) == 1
    assert loaded.tasks[0].status == "completed"
