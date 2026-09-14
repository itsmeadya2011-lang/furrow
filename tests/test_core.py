import asyncio
import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_with_no_tasks():
    o = Orchestrator(goal="test")
    assert o._is_done() is True


def test_orchestrator_is_done_when_all_completed():
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert o._is_done() is True


def test_orchestrator_is_not_done_when_tasks_pending():
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert o._is_done() is False


def test_orchestrator_is_not_done_when_task_failed():
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert o._is_done() is False


def test_orchestrator_tracks_tasks_across_cycles():
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
    ]
    assert o._is_done() is True
    assert len(o._tasks) == 1


def test_max_cycles_check():
    from furrow.config import Settings
    s = Settings(max_cycles=3)
    assert s.max_cycles == 3


def test_test_result_from_failure_json():
    t = TestResult(
        passed=False,
        summary="1 test failed",
        failures=["AssertionError: expected True"],
    )
    assert t.passed is False
    assert len(t.failures) == 1
