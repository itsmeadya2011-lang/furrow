import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_parse_empty():
    p = Plan(tasks=[], rationale="nothing to do")
    assert len(p.tasks) == 0


def test_task_status_default():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"


def test_task_result_default():
    t = TaskModel(id="1", description="do thing")
    assert t.result is None


def test_task_files_default():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []


def test_task_dependencies_default():
    t = TaskModel(id="1", description="do thing")
    assert t.dependencies == []


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_failures():
    t = TestResult(passed=False, summary="fail", failures=["error 1"])
    assert t.passed is False
    assert len(t.failures) == 1


def test_test_result_failures_default():
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []
