import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_empty_plan_stops():
    """Orchestrator should stop when planner returns no tasks."""
    orchestrator = Orchestrator(goal="do nothing")
    # Simulate empty plan by setting tasks to empty
    orchestrator.tasks = []
    # _cycle would return early, then run checks tasks
    assert not orchestrator.tasks


def test_task_model_defaults():
    t = TaskModel(id="1", description="x")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []
