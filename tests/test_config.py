import pytest
from furrow.config import (
    Provider,
    TaskModel,
    Plan,
    TestResult,
    Settings,
)


def test_provider_enum_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_task_model_default_fields():
    task = TaskModel(id="1", description="test task")
    assert task.id == "1"
    assert task.description == "test task"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_with_all_fields():
    task = TaskModel(
        id="2",
        description="complex task",
        files=["file1.py", "file2.py"],
        dependencies=["1"],
        status="completed",
        result="success",
    )
    assert task.id == "2"
    assert task.description == "complex task"
    assert task.files == ["file1.py", "file2.py"]
    assert task.dependencies == ["1"]
    assert task.status == "completed"
    assert task.result == "success"


def test_plan_model():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    plan = Plan(tasks=tasks, rationale="test rationale")
    assert len(plan.tasks) == 2
    assert plan.tasks[0].id == "1"
    assert plan.tasks[1].id == "2"
    assert plan.rationale == "test rationale"


def test_test_result_default_failures_list():
    result = TestResult(passed=True, summary="all good")
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_test_result_with_failures():
    result = TestResult(
        passed=False,
        summary="some tests failed",
        failures=["test_foo failed", "test_bar failed"],
    )
    assert result.passed is False
    assert result.summary == "some tests failed"
    assert result.failures == ["test_foo failed", "test_bar failed"]


def test_settings_test_timeout_default():
    settings = Settings()
    assert settings.test_timeout == 120


def test_settings_max_cycles_default():
    settings = Settings()
    assert settings.max_cycles == 0


def test_settings_provider_default():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC


def test_settings_ollama_base_url_default():
    settings = Settings()
    assert settings.ollama_base_url == "http://localhost:11434"


def test_settings_max_parallel_tasks_default():
    settings = Settings()
    assert settings.max_parallel_tasks == 5