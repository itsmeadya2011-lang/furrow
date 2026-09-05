import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_multiple_tasks():
    tasks = [
        TaskModel(id="1", description="first", dependencies=[]),
        TaskModel(id="2", description="second", dependencies=["1"]),
        TaskModel(id="3", description="third", dependencies=["1", "2"]),
    ]
    plan = Plan(tasks=tasks, rationale="sequential")
    assert len(plan.tasks) == 3
    assert plan.tasks[1].dependencies == ["1"]
    assert plan.tasks[2].dependencies == ["1", "2"]


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="failures found", failures=["test_a failed", "test_b failed"])
    assert t.passed is False
    assert len(t.failures) == 2
    assert t.failures[0] == "test_a failed"