"""Tests for :mod:`furrow.config`: models, status coercion and settings."""

from pathlib import Path

import pytest
from pydantic import SecretStr

import furrow.config as config_module
from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    get_settings,
    reset_settings_cache,
)


def test_task_status_values() -> None:
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"


def test_task_model_default_status_is_pending() -> None:
    task = TaskModel(id="1", description="x")
    assert task.status == TaskStatus.PENDING
    assert task.status is TaskStatus.PENDING


def test_task_model_accepts_a_plain_status_string() -> None:
    task = TaskModel(id="1", description="x", status="completed")
    assert task.status == TaskStatus.COMPLETED
    # ``TaskStatus`` subclasses ``str``, so legacy comparisons still hold.
    assert task.status == "completed"
    assert task.status.value == "completed"


def test_task_model_accepts_an_enum_status() -> None:
    task = TaskModel(id="1", description="x", status=TaskStatus.FAILED)
    assert task.status == TaskStatus.FAILED
    assert task.status == "failed"


def test_task_model_rejects_an_unknown_status_string() -> None:
    with pytest.raises(ValueError):
        TaskModel(id="1", description="x", status="bogus")


def test_task_model_rejects_a_non_string_status() -> None:
    with pytest.raises(ValueError):
        TaskModel(id="1", description="x", status=7)


def test_task_model_defaults_for_result_and_lists() -> None:
    task = TaskModel(id="1", description="x")
    assert task.result is None
    assert task.files == []
    assert task.dependencies == []


def test_task_model_list_defaults_are_not_shared_between_instances() -> None:
    first = TaskModel(id="1", description="x")
    second = TaskModel(id="2", description="y")
    assert first.files == []
    assert second.files == []
    assert first.dependencies == []
    assert second.dependencies == []

    first.files.append("a.py")
    first.dependencies.append("2")

    assert second.files == []
    assert second.dependencies == []


def test_plan_holds_tasks_and_rationale() -> None:
    plan = Plan(tasks=[{"id": "1", "description": "x"}], rationale="why")
    assert plan.rationale == "why"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].description == "x"
    assert plan.tasks[0].status == TaskStatus.PENDING


def test_test_result_defaults() -> None:
    result = TestResult(passed=True, summary="ok")
    assert result.passed is True
    assert result.summary == "ok"
    assert result.failures == []


def test_test_result_failures_default_is_not_shared() -> None:
    first = TestResult(passed=True, summary="a")
    second = TestResult(passed=True, summary="b")
    first.failures.append("boom")
    assert second.failures == []


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.provider == Provider.ANTHROPIC
    assert settings.provider.value == "anthropic"
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.max_tokens == 4096
    assert settings.test_command is None
    assert isinstance(settings.workspace, Path)


def test_settings_ignored_dirs_defaults() -> None:
    settings = Settings()
    assert isinstance(settings.ignored_dirs, list)
    assert all(isinstance(entry, str) for entry in settings.ignored_dirs)
    assert settings.ignored_dirs, "the default list must not be empty"
    assert settings.ignored_dirs[:3] == [".git", "__pycache__", "node_modules"]
    assert ".git" in settings.ignored_dirs
    assert "node_modules" in settings.ignored_dirs


def test_settings_ignored_dirs_is_a_fresh_list_per_instance() -> None:
    first = Settings()
    second = Settings()
    first.ignored_dirs.append("custom_dir")
    assert "custom_dir" in first.ignored_dirs
    assert "custom_dir" not in second.ignored_dirs
    assert second.ignored_dirs, "the untouched copy must still hold the defaults"
    assert second.ignored_dirs[:3] == [".git", "__pycache__", "node_modules"]


def test_api_keys_are_none_when_unset() -> None:
    reset_settings_cache()
    settings = get_settings()
    assert settings.anthropic_api_key is None
    assert settings.anthropic_api_key_text is None
    assert settings.openai_api_key is None
    assert settings.openai_api_key_text is None


def test_anthropic_api_key_is_a_secret_str(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "sk-test")
    reset_settings_cache()
    settings = get_settings()
    assert settings.anthropic_api_key_text == "sk-test"
    assert isinstance(settings.anthropic_api_key, SecretStr)
    # The raw field is a SecretStr wrapper, never a plain ``str``.
    assert not isinstance(settings.anthropic_api_key, str)
    assert settings.anthropic_api_key.get_secret_value() == "sk-test"


def test_openai_api_key_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_OPENAI_API_KEY", "sk-openai")
    reset_settings_cache()
    settings = get_settings()
    assert settings.openai_api_key_text == "sk-openai"
    assert isinstance(settings.openai_api_key, SecretStr)


def test_env_var_overrides_a_default(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_settings_cache()
    assert get_settings().max_tokens == 4096

    monkeypatch.setenv("FURROW_MAX_TOKENS", "1234")
    reset_settings_cache()
    assert get_settings().max_tokens == 1234


def test_get_settings_is_cached_until_reset() -> None:
    reset_settings_cache()
    first = get_settings()
    second = get_settings()
    assert isinstance(first, Settings)
    assert first is second

    reset_settings_cache()
    third = get_settings()
    assert isinstance(third, Settings)
    assert third is not first


def test_module_getattr_resolves_settings_lazily() -> None:
    reset_settings_cache()
    settings = getattr(config_module, "settings")
    assert isinstance(settings, Settings)
    assert getattr(config_module, "settings") is settings


def test_module_getattr_rejects_unknown_attributes() -> None:
    with pytest.raises(AttributeError):
        getattr(config_module, "not_a_real_config_attribute")