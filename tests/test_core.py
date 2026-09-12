import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.max_retries == 3
    assert s.request_timeout == 120.0
    assert s.max_cycle_failures == 5
    assert s.enable_file_edits is True
    assert s.max_steps == 10
    assert s.log_format == "console"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_RETRIES", "5")
    monkeypatch.setenv("FURROW_ENABLE_FILE_EDITS", "false")
    s = Settings()
    assert s.max_retries == 5
    assert s.enable_file_edits is False


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []