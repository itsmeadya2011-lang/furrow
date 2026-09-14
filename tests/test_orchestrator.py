"""Tests for the orchestrator."""
from __future__ import annotations

import pytest

from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator


def test_orchestrator_initial_state():
    """Test that orchestrator starts with empty state."""
    o = Orchestrator(goal="test goal")
    assert o.goal == "test goal"
    assert o.cycles == 0
    assert o._all_tasks == []
    assert o._completed_ids == set()
    assert o._failed_ids == set()


def test_orchestrator_is_done_empty():
    """Test that orchestrator with no tasks is done."""
    o = Orchestrator(goal="test")
    assert o._is_done() is True


def test_orchestrator_is_done_with_completed():
    """Test that orchestrator is done when all tasks completed."""
    o = Orchestrator(goal="test")
    o._all_tasks.append(TaskModel(id="1", description="task1"))
    o._completed_ids.add("1")
    assert o._is_done() is True


def test_orchestrator_is_not_done_with_failed():
    """Test that orchestrator is not done when any task failed."""
    o = Orchestrator(goal="test")
    o._all_tasks.append(TaskModel(id="1", description="task1"))
    o._completed_ids.add("1")
    o._failed_ids.add("2")
    assert o._is_done() is False


def test_topological_sort():
    """Test that tasks are sorted by dependencies."""
    o = Orchestrator(goal="test")
    t1 = TaskModel(id="1", description="first", dependencies=[])
    t2 = TaskModel(id="2", description="second", dependencies=["1"])
    t3 = TaskModel(id="3", description="third", dependencies=["1"])
    ordered = o._topological_sort([t2, t3, t1])
    # t1 should come before t2 and t3
    ids = [t.id for t in ordered]
    assert ids.index("1") < ids.index("2")
    assert ids.index("1") < ids.index("3")


def test_topological_sort_independent():
    """Test that independent tasks preserve order."""
    o = Orchestrator(goal="test")
    t1 = TaskModel(id="1", description="first", dependencies=[])
    t2 = TaskModel(id="2", description="second", dependencies=[])
    ordered = o._topological_sort([t1, t2])
    assert [t.id for t in ordered] == ["1", "2"]


def test_topological_sort_chain():
    """Test a chain of dependencies."""
    o = Orchestrator(goal="test")
    t1 = TaskModel(id="1", description="first", dependencies=[])
    t2 = TaskModel(id="2", description="second", dependencies=["1"])
    t3 = TaskModel(id="3", description="third", dependencies=["2"])
    ordered = o._topological_sort([t3, t2, t1])
    assert [t.id for t in ordered] == ["1", "2", "3"]


def test_stream_callback():
    """Test that stream callback can be set."""
    o = Orchestrator(goal="test")
    o.set_stream_callback(lambda x: None)
    assert o._stream_callback is not None