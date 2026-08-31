import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_taskmodel_defaults():
    t = TaskModel(id="1", description="x")
    assert t.status == "pending"
    assert t.retries == 0
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_settings_defaults():
    s = Settings()
    assert s.max_retries == 3
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
