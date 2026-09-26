from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_settings(tmp_path, **overrides):
    kwargs = {"workspace": tmp_path, "max_parallel_tasks": 2, "max_cycles": 0}
    kwargs.update(overrides)
    return Settings(**kwargs)


def _make_orchestrator(tmp_path, goal="test goal", **settings_overrides):
    s = _make_settings(tmp_path, **settings_overrides)
    return Orchestrator(goal=goal, client=MagicMock(), settings=s)


class _FakeWorkerAgent:
    def __init__(self, task, client=None, settings=None):
        self.task = task

    async def run(self):
        return f"result-{self.task.id}"


class _FakeTesterAgent:
    def __init__(self, client=None, settings=None):
        pass

    async def run(self, goal, tasks):
        return TestResult(passed=True, summary="ok", failures=[])


class _TimingWorkerAgent:
    def __init__(self, task, client=None, settings=None):
        self.task = task

    async def run(self):
        _started[self.task.id] = time.monotonic()
        await asyncio.sleep(0.02)
        _ended[self.task.id] = time.monotonic()
        return f"result-{self.task.id}"


_started: dict[str, float] = {}
_ended: dict[str, float] = {}


# (a) _is_done semantics
def test_is_done_empty(tmp_path):
    orch = _make_orchestrator(tmp_path)
    assert orch._is_done() is False


def test_is_done_pending(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
    assert orch._is_done() is False


def test_is_done_all_completed(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = [TaskModel(id="1", description="a", status="completed"),
                  TaskModel(id="2", description="b", status="completed")]
    assert orch._is_done() is True


def test_is_done_with_failure(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = [TaskModel(id="1", description="a", status="completed"),
                  TaskModel(id="2", description="b", status="failed")]
    assert orch._is_done() is False


# (b) max_cycles halts the loop
@pytest.mark.asyncio
async def test_max_cycles_halts(tmp_path):
    orch = _make_orchestrator(tmp_path, goal="loop me", max_cycles=3)
    orch.planner.plan = AsyncMock(
        return_value=Plan(tasks=[TaskModel(id="t1", description="do stuff")], rationale="r")
    )
    orch._is_done = lambda: False

    with patch("furrow.core.orchestrator.WorkerAgent", _FakeWorkerAgent), patch(
        "furrow.core.orchestrator.TesterAgent", _FakeTesterAgent
    ):
        await orch.run()

    assert orch.cycles == 3


# (c) dependency ordering runs a dependent task only after its dependency
@pytest.mark.asyncio
async def test_dependency_ordering(tmp_path):
    _started.clear()
    _ended.clear()
    tasks = [
        TaskModel(id="a", description="A", dependencies=[]),
        TaskModel(id="b", description="B", dependencies=["a"]),
        TaskModel(id="c", description="C", dependencies=["b"]),
    ]
    orch = _make_orchestrator(tmp_path)
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=tasks, rationale="r"))

    with patch("furrow.core.orchestrator.WorkerAgent", _TimingWorkerAgent), patch(
        "furrow.core.orchestrator.TesterAgent", _FakeTesterAgent
    ):
        await orch._cycle()

    assert orch.tasks[0].status == "completed"
    assert orch.tasks[1].status == "completed"
    assert orch.tasks[2].status == "completed"
    assert _ended["a"] < _started["b"]
    assert _ended["b"] < _started["c"]


# Persistence: state file is written and restored
@pytest.mark.asyncio
async def test_save_and_load_state(tmp_path):
    orch = _make_orchestrator(tmp_path, goal="persist me")
    orch.tasks = [TaskModel(id="t1", description="d", status="completed", result="ok")]
    orch._last_rationale = "because"
    await orch._save_state()

    restored = Orchestrator(goal="ignored", client=MagicMock(), settings=_make_settings(tmp_path))
    assert restored.goal == "persist me"
    assert restored.cycles == 0
    assert restored._last_rationale == "because"
    assert len(restored.tasks) == 1
    assert restored.tasks[0].id == "t1"
    assert restored.tasks[0].status == "completed"


def test_is_done_get_tasks_returns_self_tasks(tmp_path):
    orch = _make_orchestrator(tmp_path)
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    assert orch._get_tasks() is orch.tasks
