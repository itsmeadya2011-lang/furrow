import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_with_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="task one", files=["a.py"], dependencies=[]),
        TaskModel(id="2", description="task two", files=["b.py"], dependencies=["1"]),
    ]
    p = Plan(tasks=tasks, rationale="multi")
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


def test_task_model_defaults():
    t = TaskModel(id="x", description="desc")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="broken", failures=["fail 1", "fail 2"])
    assert t.passed is False
    assert len(t.failures) == 2
