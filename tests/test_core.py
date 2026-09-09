import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_is_done_no_tasks():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is True


def test_is_done_all_completed():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


def test_is_done_has_failures():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_is_done_has_pending():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="pending"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_get_tasks_returns_current_plan_tasks():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="ok",
    )
    orchestrator.current_plan = plan
    tasks = orchestrator._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_get_tasks_returns_empty_without_plan():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._get_tasks() == []
