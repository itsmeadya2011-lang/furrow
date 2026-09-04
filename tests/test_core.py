import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="x")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="a", files=["a.py"]),
            TaskModel(id="2", description="b", files=["b.py"], dependencies=["1"]),
        ],
        rationale="two tasks",
    )
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_failures():
    t = TestResult(passed=False, summary="boom", failures=["err1", "err2"])
    assert t.passed is False
    assert len(t.failures) == 2
