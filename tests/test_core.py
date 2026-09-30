import os

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_with_multiple_tasks():
    p = Plan(
        tasks=[
            TaskModel(id="1", description="task one"),
            TaskModel(id="2", description="task two"),
        ],
        rationale="multiple tasks",
    )
    assert len(p.tasks) == 2
    assert p.tasks[0].id == "1"
    assert p.tasks[1].id == "2"
    assert p.rationale == "multiple tasks"


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert len(p.tasks) == 0
    assert p.rationale == "nothing to do"


def test_plan_model_dump():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    dumped = p.model_dump()
    assert dumped["rationale"] == "ok"
    assert dumped["tasks"][0]["id"] == "1"
    assert dumped["tasks"][0]["description"] == "do thing"


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_task_model_with_all_fields():
    t = TaskModel(
        id="abc",
        description="implement feature",
        files=["src/main.py", "tests/test_main.py"],
        dependencies=["dep-1"],
        status="completed",
        result="done",
    )
    assert t.id == "abc"
    assert t.files == ["src/main.py", "tests/test_main.py"]
    assert t.dependencies == ["dep-1"]
    assert t.status == "completed"
    assert t.result == "done"


def test_task_model_requires_id():
    with pytest.raises(ValidationError):
        TaskModel(description="no id")


def test_task_model_requires_description():
    with pytest.raises(ValidationError):
        TaskModel(id="1")


def test_test_result_defaults():
    t = TestResult(passed=True, summary="all good")
    assert t.passed is True
    assert t.summary == "all good"
    assert t.failures == []


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="some tests failed", failures=["test_a", "test_b"])
    assert t.passed is False
    assert len(t.failures) == 2
    assert "test_a" in t.failures


def test_provider_enum_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_provider_enum_from_string():
    assert Provider("anthropic") == Provider.ANTHROPIC
    assert Provider("openai") == Provider.OPENAI
    assert Provider("ollama") == Provider.OLLAMA


def test_provider_enum_count():
    assert len(Provider) == 3


def test_settings_defaults(monkeypatch):
    for key in list(os.environ):
        if key.startswith("FURROW_"):
            monkeypatch.delenv(key, raising=False)
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.planner_model == "claude-3-5-haiku-20241022"
    assert s.worker_model == "claude-3-5-sonnet-20241022"
    assert s.tester_model == "claude-3-5-sonnet-20241022"
    assert s.anthropic_api_key is None
    assert s.openai_api_key is None
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.log_level == "INFO"


def test_settings_custom_values():
    s = Settings(
        provider=Provider.OPENAI,
        model="gpt-4-turbo",
        planner_model="gpt-4o-mini",
        worker_model="gpt-4-turbo",
        tester_model="gpt-4o-mini",
        openai_api_key="my-openai-key",
        ollama_base_url="http://localhost:8080",
        max_parallel_tasks=3,
        max_cycles=10,
        log_level="DEBUG",
    )
    assert s.provider == Provider.OPENAI
    assert s.model == "gpt-4-turbo"
    assert s.planner_model == "gpt-4o-mini"
    assert s.openai_api_key == "my-openai-key"
    assert s.ollama_base_url == "http://localhost:8080"
    assert s.max_parallel_tasks == 3
    assert s.max_cycles == 10
    assert s.log_level == "DEBUG"


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MODEL", "gpt-4")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "4")
    monkeypatch.setenv("FURROW_OPENAI_API_KEY", "env-key")
    for key in list(os.environ):
        if key.startswith("FURROW_") and key not in (
            "FURROW_PROVIDER",
            "FURROW_MODEL",
            "FURROW_MAX_CYCLES",
            "FURROW_MAX_PARALLEL_TASKS",
            "FURROW_OPENAI_API_KEY",
        ):
            monkeypatch.delenv(key, raising=False)
    s = Settings()
    assert s.provider == Provider.OPENAI
    assert s.model == "gpt-4"
    assert s.max_cycles == 7
    assert s.max_parallel_tasks == 4
    assert s.openai_api_key == "env-key"


def test_settings_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    s = Settings(provider=Provider.OLLAMA, anthropic_api_key="test")
    assert s.provider == Provider.OLLAMA
