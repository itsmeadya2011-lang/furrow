"""Tests for :mod:`furrow.config`.

The settings layer is a public contract: the CLI mutates the cached instance, the web layer reads
``workspace``, and every agent reads model names and timeouts from it. These tests pin the
documented defaults, the ``FURROW_`` env prefix, the cache lifecycle, and the validation bounds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import furrow.config
from furrow.config import (
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    get_settings,
    reset_settings,
)
from tests.conftest import build_settings

# --------------------------------------------------------------------------------------
# TaskStatus / TaskModel
# --------------------------------------------------------------------------------------


def test_task_status_is_a_str_enum_so_raw_strings_still_compare_equal() -> None:
    assert issubclass(TaskStatus, str)
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.FAILED == "failed"


def test_new_task_defaults_to_pending_and_compares_equal_to_the_string() -> None:
    task = TaskModel(id="1", description="do thing")
    assert task.status == "pending"
    assert task.status is TaskStatus.PENDING


def test_task_status_serialises_as_a_plain_string_for_json_consumers() -> None:
    task = TaskModel(id="1", description="do thing", status="failed")  # type: ignore[arg-type]
    assert json.dumps({"status": task.status}) == '{"status": "failed"}'
    assert json.loads('{"status": "skipped"}') == {"status": "skipped"}


def test_assigning_a_raw_status_string_is_accepted_by_the_orchestrator() -> None:
    # furrow/core/orchestrator.py assigns literals such as task.status = "completed"; pydantic
    # must not reject them at runtime.
    task = TaskModel(id="1", description="do thing")
    task.status = "completed"  # type: ignore[assignment]
    assert task.status == "completed"


def test_unknown_status_value_is_rejected_at_construction() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="do thing", status="complete")  # type: ignore[arg-type]


def test_optional_task_fields_default_to_empty_and_result_to_none() -> None:
    task = TaskModel(id="1", description="do thing")
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_test_result_failures_default_to_an_independent_list() -> None:
    first = TestResult(passed=False, summary="a")
    second = TestResult(passed=False, summary="b")
    first.failures.append("boom")
    assert second.failures == []


# --------------------------------------------------------------------------------------
# Documented defaults
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("provider", Provider.ANTHROPIC),
        ("model", "claude-sonnet-4-20250514"),
        ("planner_model", "claude-3-5-haiku-20241022"),
        ("worker_model", "claude-3-5-sonnet-20241022"),
        ("tester_model", "claude-3-5-sonnet-20241022"),
        ("anthropic_api_key", None),
        ("openai_api_key", None),
        ("ollama_base_url", "http://localhost:11434"),
        ("max_parallel_tasks", 5),
        ("max_cycles", 0),
        ("log_level", "INFO"),
        ("max_tokens", 8192),
        ("request_timeout", 120.0),
        ("max_retries", 3),
        ("test_timeout", 300.0),
    ],
)
def test_field_default_matches_the_documented_contract(name: str, expected: object) -> None:
    settings = build_settings()
    assert getattr(settings, name) == expected


def test_workspace_default_is_the_current_directory() -> None:
    assert build_settings().workspace == Path.cwd()


def test_max_cycles_zero_documents_unlimited() -> None:
    assert "unlimited" in (Settings.model_fields["max_cycles"].description or "").lower()


# --------------------------------------------------------------------------------------
# Environment / .env plumbing
# --------------------------------------------------------------------------------------


def test_furrow_prefixed_environment_variable_is_applied(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_TOKENS", "1234")
    assert build_settings().max_tokens == 1234


def test_environment_variable_without_the_prefix_is_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_TOKENS", "1234")
    assert build_settings().max_tokens == 8192


def test_prefixed_environment_variable_beats_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_TEST_TIMEOUT", "12.5")
    settings = build_settings()
    assert settings.max_parallel_tasks == 9
    assert settings.provider is Provider.OPENAI
    assert settings.test_timeout == 12.5


def test_env_file_is_loaded_when_one_is_configured(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("FURROW_LOG_LEVEL=DEBUG\nFURROW_WORKSPACE=/tmp/from-env\n")
    settings = build_settings(_env_file=env_file)
    assert settings.log_level == "DEBUG"
    assert settings.workspace == Path("/tmp/from-env")


def test_settings_declare_dot_env_as_the_default_env_file() -> None:
    assert Settings.model_config.get("env_file") == ".env"
    assert Settings.model_config.get("env_prefix") == "FURROW_"


def test_explicit_argument_wins_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_RETRIES", "9")
    assert build_settings(max_retries=1).max_retries == 1


# --------------------------------------------------------------------------------------
# get_settings / reset_settings cache lifecycle
# --------------------------------------------------------------------------------------


def test_get_settings_builds_once_and_caches() -> None:
    first = get_settings()
    assert first is get_settings()
    assert isinstance(first, Settings)


def test_reset_settings_forces_a_reread_of_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MODEL", "first-model")
    first = get_settings()
    assert first.model == "first-model"

    monkeypatch.setenv("FURROW_MODEL", "second-model")
    assert get_settings() is first, "cached settings must not change under the caller"
    assert get_settings().model == "first-model"

    reset_settings()
    second = get_settings()
    assert second is not first
    assert second.model == "second-model"


def test_overrides_rebuild_and_recache_the_settings() -> None:
    baseline = get_settings()
    overridden = get_settings(model="override-model", max_cycles=3)
    assert overridden is not baseline
    assert overridden.model == "override-model"
    assert overridden.max_cycles == 3
    # The override sticks: this is what furrow/cli/main.py's lazy ``settings`` attribute relies on.
    assert get_settings() is overridden
    assert get_settings().model == "override-model"


def test_overrides_do_not_leak_unrelated_environment_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_LOG_LEVEL", "ERROR")
    assert get_settings(log_level="WARNING").log_level == "WARNING"
    reset_settings()
    assert get_settings().log_level == "ERROR"


def test_overrides_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        get_settings(max_retries=-1)
    reset_settings()
    assert get_settings().max_retries == 3


# --------------------------------------------------------------------------------------
# Module-level lazy ``settings`` attribute
# --------------------------------------------------------------------------------------


def test_lazy_module_attribute_is_the_cached_settings() -> None:
    assert furrow.config.settings is get_settings()


def test_mutating_the_lazy_attribute_is_visible_to_later_readers() -> None:
    furrow.config.settings.model = "mutated"
    assert get_settings().model == "mutated"


def test_unknown_module_attribute_still_raises_attribute_error() -> None:
    missing = "definitely_not_a_setting"
    with pytest.raises(AttributeError):
        getattr(furrow.config, missing)


def test_dir_advertises_the_lazy_settings_attribute() -> None:
    assert "settings" in dir(furrow.config)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------


def test_max_retries_below_zero_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_settings(max_retries=-1)


def test_max_retries_zero_is_allowed_because_it_means_one_attempt() -> None:
    assert build_settings(max_retries=0).max_retries == 0


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_settings(provider="not-a-provider")


@pytest.mark.parametrize("provider", ["anthropic", "openai", "ollama"])
def test_every_documented_provider_is_accepted(provider: str) -> None:
    assert build_settings(provider=provider).provider == Provider(provider)


def test_non_numeric_timeout_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_settings(request_timeout="soon")


def test_non_numeric_max_tokens_is_rejected() -> None:
    with pytest.raises(ValidationError):
        build_settings(max_tokens="lots")


def test_workspace_string_is_coerced_to_a_path(tmp_path: Path) -> None:
    settings = build_settings(workspace=str(tmp_path / "ws"))  # type: ignore[arg-type]
    assert isinstance(settings.workspace, Path)
    assert settings.workspace == tmp_path / "ws"
