"""Unit tests for `furrow.config`: enums, models, env parsing and the settings singleton."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    get_settings,
    reload_settings,
)
from furrow.config import settings as singleton


def test_task_status_has_exactly_the_four_lifecycle_members() -> None:
    assert [member.name for member in TaskStatus] == [
        "PENDING",
        "IN_PROGRESS",
        "COMPLETED",
        "FAILED",
    ]


def test_task_status_values() -> None:
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.IN_PROGRESS.value == "in_progress"
    assert TaskStatus.COMPLETED.value == "completed"
    assert TaskStatus.FAILED.value == "failed"


def test_task_status_is_a_string_enum() -> None:
    # The orchestrator and the web API both serialise these values as bare strings.
    assert TaskStatus.PENDING == "pending"
    assert isinstance(TaskStatus.PENDING, str)
    assert TaskStatus("failed") is TaskStatus.FAILED


def test_provider_values() -> None:
    assert [provider.value for provider in Provider] == ["anthropic", "openai", "ollama"]


def test_provider_is_a_string_enum() -> None:
    assert Provider.OLLAMA == "ollama"
    assert Provider("openai") is Provider.OPENAI


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="write the thing")
    assert task.id == "1"
    assert task.description == "write the thing"
    assert task.files == []
    assert task.dependencies == []
    assert task.status is TaskStatus.PENDING
    assert task.result is None
    assert task.attempts == 0


def test_task_model_mutable_defaults_are_not_shared() -> None:
    first = TaskModel(id="1", description="a")
    second = TaskModel(id="2", description="b")
    first.files.append("a.py")
    first.dependencies.append("other")
    assert second.files == []
    assert second.dependencies == []


@pytest.mark.parametrize("payload", [{}, {"id": "1"}, {"description": "d"}])
def test_task_model_requires_id_and_description(payload: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TaskModel(**payload)


def test_task_model_rejects_a_numeric_id() -> None:
    # The planner normalises ids to strings; the model itself must not silently coerce.
    with pytest.raises(ValidationError):
        TaskModel(id=1, description="d")  # type: ignore[arg-type]


def test_task_model_accepts_a_status_string() -> None:
    task = TaskModel(id="1", description="d", status="completed")  # type: ignore[arg-type]
    assert task.status is TaskStatus.COMPLETED


def test_task_model_tracks_attempts_and_result() -> None:
    task = TaskModel(id="1", description="d", attempts=3, result="did it")
    assert (task.attempts, task.result) == (3, "did it")


def test_plan_requires_a_tasks_array() -> None:
    with pytest.raises(ValidationError):
        Plan()  # type: ignore[call-arg]


def test_plan_rationale_defaults_to_empty_string() -> None:
    # Deliberate robustness change: a plan without a rationale must not raise.
    plan = Plan(tasks=[TaskModel(id="1", description="d")])
    assert plan.rationale == ""


def test_plan_round_trips_tasks_and_rationale() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="d")], rationale="because")
    rebuilt = Plan.model_validate(plan.model_dump())
    assert rebuilt == plan
    assert rebuilt.tasks[0].id == "1"


def test_test_result_defaults() -> None:
    result = TestResult(passed=True)
    assert result.passed is True
    assert result.summary == ""
    assert result.failures == []
    assert result.commands == []


def test_test_result_requires_passed() -> None:
    with pytest.raises(ValidationError):
        TestResult()  # type: ignore[call-arg]


def test_test_result_is_not_collected_by_pytest() -> None:
    # `__test__ = False` is what stops pytest warning about a class named TestResult.
    assert TestResult.__test__ is False


def test_test_result_round_trips_including_commands() -> None:
    result = TestResult(
        passed=False,
        summary="1 failed",
        failures=["test_x"],
        commands=["python3 -m pytest -q"],
    )
    assert TestResult.model_validate(result.model_dump()) == result


def test_settings_reads_the_furrow_env_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    assert reload_settings().max_cycles == 7


def test_settings_ignores_unprefixed_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_CYCLES", "7")
    assert reload_settings().max_cycles == 0


def test_settings_ignores_the_prefix_without_the_env_var(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("FURROW_MAX_CYCLES", raising=False)
    assert reload_settings().max_cycles == 0


def test_settings_parses_an_enum_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    assert reload_settings().provider is Provider.OLLAMA


def test_settings_needs_no_api_key_to_construct() -> None:
    configured = reload_settings(anthropic_api_key=None, openai_api_key=None)
    assert configured.anthropic_api_key is None
    assert configured.openai_api_key is None


def test_settings_workspace_defaults_to_cwd() -> None:
    assert reload_settings().workspace == Path.cwd()


def test_settings_state_file_defaults_to_none() -> None:
    assert reload_settings().state_file is None


def test_settings_loop_bounds_defaults() -> None:
    configured = reload_settings()
    assert configured.max_cycles == 0  # 0 means "run until a halt condition fires"
    assert configured.max_parallel_tasks == 5
    assert configured.max_stalled_cycles == 3
    assert configured.max_tool_iterations == 12
    assert configured.max_event_history == 500


def test_settings_llm_defaults() -> None:
    configured = reload_settings()
    assert configured.max_tokens == 8192
    assert configured.temperature == 0.0
    assert configured.max_retries == 3
    assert configured.retry_base_delay == 1.0
    assert configured.request_timeout == 120.0
    assert configured.ollama_base_url == "http://localhost:11434"


def test_settings_web_defaults() -> None:
    configured = reload_settings()
    assert configured.web_host == "127.0.0.1"
    assert configured.web_port == 8000


def test_settings_exclude_dirs_defaults_cover_the_usual_noise() -> None:
    configured = reload_settings()
    for noisy in (".git", ".venv", "node_modules", "__pycache__", "*.egg-info"):
        assert noisy in configured.exclude_dirs


def test_settings_exclude_dirs_default_is_not_shared_between_instances() -> None:
    first = reload_settings()
    first.exclude_dirs.append("my-private-cache")
    assert "my-private-cache" not in reload_settings().exclude_dirs


def test_api_key_for_returns_the_configured_anthropic_key() -> None:
    configured = reload_settings(anthropic_api_key="sk-anthropic")
    assert configured.api_key_for(Provider.ANTHROPIC) == "sk-anthropic"


def test_api_key_for_falls_back_to_the_bare_anthropic_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bare-anthropic")
    assert reload_settings().api_key_for(Provider.ANTHROPIC) == "bare-anthropic"


def test_api_key_for_falls_back_to_the_bare_openai_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "bare-openai")
    assert reload_settings().api_key_for(Provider.OPENAI) == "bare-openai"


def test_api_key_for_prefers_the_configured_key_over_the_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "bare-anthropic")
    configured = reload_settings(anthropic_api_key="configured")
    assert configured.api_key_for(Provider.ANTHROPIC) == "configured"


def test_api_key_for_defaults_to_the_active_provider() -> None:
    configured = reload_settings(
        provider=Provider.OPENAI,
        anthropic_api_key="sk-anthropic",
        openai_api_key="sk-openai",
    )
    assert configured.api_key_for() == "sk-openai"


def test_api_key_for_ollama_is_none_without_a_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Ollama needs no credentials, so a missing key is not a configuration error.
    monkeypatch.setenv("OPENAI_API_KEY", "bare-openai")
    assert reload_settings().api_key_for(Provider.OLLAMA) is None


def test_api_key_for_is_none_when_nothing_is_configured() -> None:
    configured = reload_settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)
    assert configured.api_key_for() is None


def test_resolved_workspace_returns_the_resolved_existing_path(tmp_path: Path) -> None:
    nested = tmp_path / "repo"
    nested.mkdir()
    assert reload_settings(workspace=nested).resolved_workspace() == nested.resolve()


def test_resolved_workspace_falls_back_to_cwd_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "gone"
    assert not missing.exists()
    assert reload_settings(workspace=missing).resolved_workspace() == Path.cwd()


def test_get_settings_returns_the_process_singleton() -> None:
    assert get_settings() is singleton


def test_reload_settings_does_not_mutate_the_singleton(tmp_path: Path) -> None:
    before = get_settings()
    replaced = reload_settings(workspace=tmp_path, max_cycles=99, model="other-model")
    assert replaced.max_cycles == 99
    assert replaced is not before
    assert get_settings() is before
    assert before.max_cycles != 99
    assert before.workspace != tmp_path


def test_reload_settings_returns_independent_instances() -> None:
    first = reload_settings(max_cycles=3)
    second = reload_settings(max_cycles=4)
    assert (first.max_cycles, second.max_cycles) == (3, 4)


def test_reload_settings_returns_a_settings_instance(tmp_path: Path) -> None:
    assert isinstance(reload_settings(), Settings)
