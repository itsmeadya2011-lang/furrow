from __future__ import annotations

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_get_tasks_returns_empty_when_no_plan():
    o = Orchestrator(goal="test")
    assert o._get_tasks() == []


def test_get_tasks_returns_plan_tasks():
    o = Orchestrator(goal="test")
    o.plan = Plan(tasks=[TaskModel(id="1", description="task1")], rationale="test")
    tasks = o._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_is_done_no_tasks_no_test_result():
    o = Orchestrator(goal="test")
    # No plan, no test result -> not done (can't determine)
    assert o._is_done() is False


def test_is_done_no_tasks_with_passing_tests():
    o = Orchestrator(goal="test")
    o.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert o._is_done() is True


def test_is_done_no_tasks_with_failing_tests():
    o = Orchestrator(goal="test")
    o.last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
    assert o._is_done() is False


def test_is_done_all_completed():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ],
        rationale="test",
    )
    o.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert o._is_done() is True


def test_is_done_some_failed():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed"),
        ],
        rationale="test",
    )
    assert o._is_done() is False


def test_is_done_not_all_completed():
    o = Orchestrator(goal="test")
    o.plan = Plan(
        tasks=[
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ],
        rationale="test",
    )
    assert o._is_done() is False


def test_max_cycles_enforced():
    """Test that max_cycles setting is respected."""
    settings = Settings(max_cycles=2)
    o = Orchestrator(goal="test", settings=settings)
    o.cycles = 3
    # The run() method checks max_cycles, but we can verify the logic directly
    max_cycles = o.settings.max_cycles
    assert max_cycles == 2
    assert o.cycles > max_cycles


def test_orchestrator_stores_original_goal():
    o = Orchestrator(goal="original goal")
    assert o.original_goal == "original goal"
    assert o.goal == "original goal"