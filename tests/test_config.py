import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_task_model_defaults():
    task = TaskModel(id="1", description="test task")
    assert task.id == "1"
    assert task.description == "test task"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_plan_creation():
    task1 = TaskModel(id="1", description="first task", files=["src/test.py"])
    task2 = TaskModel(id="2", description="second task", dependencies=["1"])
    plan = Plan(tasks=[task1, task2], rationale="Test plan rationale")
    assert len(plan.tasks) == 2
    assert plan.rationale == "Test plan rationale"
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].dependencies == ["1"]


def test_test_result_defaults():
    result = TestResult(passed=True, summary="All tests passed")
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


def test_settings_defaults():
    settings = Settings(_env_file=None)
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


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MODEL", "gpt-4")
    monkeypatch.setenv("FURROW_PLANNER_MODEL", "gpt-3.5-turbo")
    monkeypatch.setenv("FURROW_WORKER_MODEL", "gpt-4")
    monkeypatch.setenv("FURROW_TESTER_MODEL", "gpt-4")
    monkeypatch.setenv("FURROW_OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "10")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "5")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)
    assert settings.provider == Provider.OPENAI
    assert settings.model == "gpt-4"
    assert settings.planner_model == "gpt-3.5-turbo"
    assert settings.worker_model == "gpt-4"
    assert settings.tester_model == "gpt-4"
    assert settings.openai_api_key == "test-key"
    assert settings.max_parallel_tasks == 10
    assert settings.max_cycles == 5
    assert settings.log_level == "DEBUG"


def test_provider_enum():
    assert Provider.ANTHROPIC.value == "anthropic"
    assert Provider.OPENAI.value == "openai"
    assert Provider.OLLAMA.value == "ollama"
    assert len(list(Provider)) == 3
