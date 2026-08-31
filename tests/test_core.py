"""Core tests for Furrow configuration models."""

import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_default_status():
    task = TaskModel(id="1", description="test")
    assert task.status == "pending"


def test_task_model_status_values():
    task = TaskModel(id="1", description="test")
    valid_statuses = ["pending", "running", "completed", "failed"]
    for status in valid_statuses:
        task.status = status
        assert task.status == status


def test_task_model_with_files():
    task = TaskModel(id="1", description="test", files=["src/main.py", "tests/test_main.py"])
    assert len(task.files) == 2
    assert "src/main.py" in task.files


def test_task_model_with_dependencies():
    task = TaskModel(id="2", description="test", dependencies=["1"])
    assert task.dependencies == ["1"]


def test_plan_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="first"),
        TaskModel(id="2", description="second"),
    ]
    plan = Plan(tasks=tasks, rationale="Two tasks")
    assert len(plan.tasks) == 2
    assert plan.rationale == "Two tasks"


def test_test_result_with_failures():
    result = TestResult(
        passed=False,
        summary="3 tests failed",
        failures=["test_auth failed", "test_db failed", "test_api failed"],
    )
    assert result.passed is False
    assert len(result.failures) == 3


def test_test_result_empty_failures():
    result = TestResult(passed=True, summary="All passed")
    assert result.failures == []
