"""Tests for the orchestrator task tracking and completion logic."""

from __future__ import annotations

import pytest
from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


class _FakeSettings(Settings):
    """Settings with a disabled API key so we can construct an orchestrator without real credentials."""

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None


def _make_orchestrator(goal: str = "test goal") -> Orchestrator:
    settings = _FakeSettings()
    return Orchestrator(goal=goal, settings=settings)


def test_orchestrator_initial_state():
    o = _make_orchestrator()
    assert o.cycles == 0
    assert o._tasks == {}
    assert o._all_completed is False


def test_is_done_initially_false():
    o = _make_orchestrator()
    assert o._is_done() is False


def test_all_tracked_tasks_done_empty():
    o = _make_orchestrator()
    assert o._all_tracked_tasks_done() is False


def test_task_merging_preserves_status():
    o = _make_orchestrator()
    t1 = TaskModel(id="1", description="old desc", files=["a.py"], status="completed", result="done")
    o._tasks["1"] = t1

    # Simulate a new plan with same task id but updated description
    new_task = TaskModel(id="1", description="new desc", files=["b.py"])
    if new_task.id not in o._tasks:
        o._tasks[new_task.id] = new_task
    else:
        existing = o._tasks[new_task.id]
        existing.description = new_task.description
        existing.files = new_task.files
        existing.dependencies = new_task.dependencies

    assert o._tasks["1"].description == "new desc"
    assert o._tasks["1"].files == ["b.py"]
    assert o._tasks["1"].status == "completed"  # preserved
    assert o._tasks["1"].result == "done"  # preserved


def test_is_done_when_all_completed():
    o = _make_orchestrator()
    o._tasks["1"] = TaskModel(id="1", description="done", status="completed")
    o._tasks["2"] = TaskModel(id="2", description="done", status="completed")
    assert o._is_done() is True


def test_is_not_done_with_pending():
    o = _make_orchestrator()
    o._tasks["1"] = TaskModel(id="1", description="pending", status="pending")
    assert o._is_done() is False


def test_is_not_done_with_failed():
    o = _make_orchestrator()
    o._tasks["1"] = TaskModel(id="1", description="failed", status="failed")
    assert o._is_done() is False


def test_all_completed_flag_short_circuits():
    o = _make_orchestrator()
    o._all_completed = True
    assert o._is_done() is True