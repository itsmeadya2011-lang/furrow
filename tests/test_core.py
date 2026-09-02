"""Tests for the core data models and basic config."""

import pytest
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_default_fields():
    p = Plan(tasks=[], rationale="empty plan")
    assert p.tasks == []
    assert p.rationale == "empty plan"


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_files():
    t = TaskModel(
        id="2", description="build",
        files=["a.py", "b.py"], dependencies=["1"], status="completed",
        result="Done"
    )
    assert t.files == ["a.py", "b.py"]
    assert t.dependencies == ["1"]
    assert t.status == "completed"
    assert t.result == "Done"


def test_task_model_serialization():
    t = TaskModel(id="1", description="test", files=["x.py"], status="completed", result="ok")
    d = t.model_dump()
    assert d["id"] == "1"
    assert d["description"] == "test"
    assert d["files"] == ["x.py"]
    assert d["status"] == "completed"
    assert d["result"] == "ok"


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="broken", failures=["test_x failed"])
    assert t.passed is False
    assert "test_x failed" in t.failures


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"
    assert s.test_timeout == 120


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    s = Settings()
    assert s.log_level == "DEBUG"
    assert s.max_cycles == 10
