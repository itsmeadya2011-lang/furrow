import pytest
from rich.console import Console

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_settings_defaults():
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.max_tokens == 4096
    assert settings.max_parallel_tasks == 5
    assert settings.ollama_base_url == "http://localhost:11434"


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_task_model():
    task = TaskModel(id="1", description="do thing")
    assert task.id == "1"
    assert task.description == "do thing"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_plan_model():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "ok"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []
