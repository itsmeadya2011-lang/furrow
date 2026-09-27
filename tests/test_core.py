import pytest
from pydantic import ValidationError

from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_with_dependencies():
    t = TaskModel(id="1", description="do thing", dependencies=["2", "3"])
    assert t.dependencies == ["2", "3"]
    assert t.status == "pending"
    assert t.result is None


def test_test_result_with_failures():
    failures = ["test_a failed", "test_b failed"]
    t = TestResult(passed=False, summary="2 failed", failures=failures)
    assert t.passed is False
    assert t.failures == failures


def test_task_model_requires_id():
    with pytest.raises(ValidationError):
        TaskModel(description="no id")
