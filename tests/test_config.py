import pytest
from furrow.config import (
    Provider,
    TaskModel,
    Plan,
    TestResult,
    Settings,
)


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_task_model_defaults():
    task = TaskModel(id="1", description="Test task")
    assert task.id == "1"
    assert task.description == "Test task"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_fields():
    task = TaskModel(
        id="2",
        description="Test with files",
        files=["src/test.py", "tests/test_test.py"],
        dependencies=["1"],
        status="completed",
        result="Done",
    )
    assert task.files == ["src/test.py", "tests/test_test.py"]
    assert task.dependencies == ["1"]
    assert task.status == "completed"
    assert task.result == "Done"


def test_plan_model():
    tasks = [TaskModel(id="1", description="Task 1")]
    plan = Plan(tasks=tasks, rationale="Test plan")
    assert len(plan.tasks) == 1
    assert plan.rationale == "Test plan"


def test_test_result_passed():
    result = TestResult(passed=True, summary="All good", failures=[])
    assert result.passed is True
    assert result.summary == "All good"
    assert result.failures == []


def test_test_result_failed():
    result = TestResult(passed=False, summary="Failed", failures=["Error 1", "Error 2"])
    assert result.passed is False
    assert result.failures == ["Error 1", "Error 2"]


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.model == "claude-3-5-sonnet-20241022"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"
    assert settings.ollama_model == "llama3.1"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.log_level == "INFO"


def test_settings_env_override(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MODEL", "gpt-4o")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "10")
    settings = Settings()
    assert settings.provider == Provider.OPENAI
    assert settings.model == "gpt-4o"
    assert settings.max_cycles == 10