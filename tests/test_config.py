from __future__ import annotations

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_task_model_defaults():
    task = TaskModel(id="1", description="do thing")
    assert task.id == "1"
    assert task.description == "do thing"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_values():
    task = TaskModel(
        id="2",
        description="write code",
        files=["main.py"],
        dependencies=["1"],
        status="in_progress",
        result="started",
    )
    assert task.files == ["main.py"]
    assert task.dependencies == ["1"]
    assert task.status == "in_progress"
    assert task.result == "started"


def test_plan_creation_with_tasks():
    tasks = [
        TaskModel(id="1", description="step one"),
        TaskModel(id="2", description="step two"),
    ]
    plan = Plan(tasks=tasks, rationale="build feature")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].description == "step one"
    assert plan.tasks[1].description == "step two"
    assert plan.rationale == "build feature"


def test_plan_empty_tasks():
    plan = Plan(tasks=[], rationale="nothing to do")
    assert plan.tasks == []
    assert plan.rationale == "nothing to do"


def test_test_result_creation():
    result = TestResult(passed=True, summary="all good", failures=[])
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_test_result_with_failures():
    result = TestResult(
        passed=False, summary="tests failed", failures=["test_a failed", "test_b failed"]
    )
    assert result.passed is False
    assert len(result.failures) == 2
    assert "test_a failed" in result.failures


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.log_level == "INFO"


def test_settings_from_env_vars(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MODEL", "gpt-4")
    monkeypatch.setenv("FURROW_OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "8")

    settings = Settings()
    assert settings.provider == Provider.OPENAI
    assert settings.model == "gpt-4"
    assert settings.openai_api_key == "sk-test-key"
    assert settings.anthropic_api_key == "sk-ant-test"
    assert settings.max_cycles == 10
    assert settings.log_level == "DEBUG"
    assert settings.max_parallel_tasks == 8


def test_settings_invalid_provider():
    with pytest.raises(ValidationError):
        Settings(provider="invalid_provider")
