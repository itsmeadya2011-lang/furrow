import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_dependencies_default_empty():
    t = TaskModel(id="1", description="task")
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_files_default_empty():
    t = TaskModel(id="1", description="task")
    assert t.files == []


def test_test_result_failures_default_empty():
    t = TestResult(passed=False, summary="fail")
    assert t.failures == []


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert p.tasks == []
    assert p.rationale == "nothing to do"