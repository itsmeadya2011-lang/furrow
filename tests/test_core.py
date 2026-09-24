import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model():
    task = TaskModel(id="1", description="test task", files=["a.py"], dependencies=["2"])
    assert task.id == "1"
    assert task.description == "test task"
    assert task.files == ["a.py"]
    assert task.dependencies == ["2"]
    assert task.status == "pending"
    assert task.result is None


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    plan = Plan(tasks=tasks, rationale="test rationale")
    assert len(plan.tasks) == 2
    assert plan.rationale == "test rationale"


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="failed", failures=["error 1", "error 2"])
    assert t.passed is False
    assert len(t.failures) == 2
    assert t.failures[0] == "error 1"