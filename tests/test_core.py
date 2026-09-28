"""Core tests for Plan and TestResult models — backward compatibility."""
import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_plan_with_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="first"),
            TaskModel(id="2", description="second", dependencies=["1"]),
        ],
        rationale="multi",
    )
    assert len(p.tasks) == 2
    assert p.tasks[1].dependencies == ["1"]


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 failed", failures=["err1", "err2"])
    assert t.passed is False
    assert len(t.failures) == 2
