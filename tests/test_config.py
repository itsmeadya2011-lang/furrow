"""Configuration and model validation.

These lock in the guarantees the rest of the loop is built on: a bad configuration
fails at startup with a clear message instead of at the first request, a
zero-task plan is legal (the orchestrator reads it as "nothing to do"), and importing
``furrow`` never requires a credential.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from furrow import config
from furrow.config import Plan, Provider, Settings, TaskModel, get_settings

# Aliased: a module-level name starting with ``Test`` makes pytest try to collect the
# pydantic model as a test class and warn about its ``__init__``.
from furrow.config import TestResult as Verdict

KNOWN_STATUSES = ("pending", "running", "completed", "failed")


def settings_from(**overrides: Any) -> Settings:
    """Build a :class:`Settings` from untyped kwargs.

    Half of this module asserts that pydantic *rejects* a value, and a typed call would
    be turned down by mypy before it ever reached a validator. The environment fixture
    in ``conftest`` has already cleared ``FURROW_*`` and pointed the CWD at a temp
    directory, so nothing else leaks in here.
    """
    return Settings(**overrides)


def _duplicate_id_plan() -> Plan:
    return Plan(
        tasks=[
            TaskModel(id="1", description="first"),
            TaskModel(id="1", description="second"),
        ],
        rationale="two tasks share an id",
    )


def test_zero_parallel_tasks_is_rejected() -> None:
    with pytest.raises(ValidationError):
        settings_from(max_parallel_tasks=0)


def test_negative_parallel_tasks_is_rejected() -> None:
    with pytest.raises(ValidationError):
        settings_from(max_parallel_tasks=-3)


def test_one_parallel_task_is_the_lowest_accepted_value() -> None:
    assert Settings(max_parallel_tasks=1).max_parallel_tasks == 1


def test_negative_max_cycles_is_rejected() -> None:
    with pytest.raises(ValidationError):
        settings_from(max_cycles=-1)


def test_zero_max_cycles_means_unlimited() -> None:
    assert Settings(max_cycles=0).max_cycles == 0


def test_unknown_log_level_is_rejected() -> None:
    with pytest.raises(ValidationError):
        settings_from(log_level="CHATTY")


@pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
def test_known_log_levels_are_accepted(log_level: str) -> None:
    assert settings_from(log_level=log_level).log_level == log_level


@pytest.mark.parametrize("field_name", ["model", "planner_model", "worker_model", "tester_model"])
def test_blank_model_id_is_rejected(field_name: str) -> None:
    with pytest.raises(ValidationError) as excinfo:
        settings_from(**{field_name: "   "})
    assert field_name in str(excinfo.value)


def test_blank_model_id_error_names_every_offending_field() -> None:
    with pytest.raises(ValidationError) as excinfo:
        settings_from(model="", tester_model="")
    assert "model, tester_model" in str(excinfo.value)


def test_blank_test_command_is_rejected() -> None:
    with pytest.raises(ValidationError) as excinfo:
        settings_from(test_command="   ")
    assert "test_command" in str(excinfo.value)


def test_omitted_test_command_means_autodetect() -> None:
    assert Settings(test_command=None).test_command is None


def test_settings_construct_with_no_api_keys() -> None:
    settings = Settings()
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None


def test_import_time_singleton_is_credential_free() -> None:
    assert config.settings.anthropic_api_key is None
    assert config.settings.openai_api_key is None


def test_removed_ollama_provider_is_rejected_at_validation() -> None:
    with pytest.raises(ValidationError) as excinfo:
        settings_from(provider="ollama")
    assert "ollama" in str(excinfo.value)


def test_ollama_is_not_a_provider_member() -> None:
    assert "ollama" not in {member.value for member in Provider}


@pytest.mark.parametrize("name", ["anthropic", "openai"])
def test_supported_providers_are_accepted(name: str) -> None:
    assert settings_from(provider=name).provider is Provider(name)


def test_env_prefix_is_furrow(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")
    assert Settings().max_parallel_tasks == 3


def test_unrelated_furrow_variables_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_SOMETHING_ELSE", "irrelevant")
    assert Settings().max_parallel_tasks == 5


def test_workspace_defaults_to_the_current_directory() -> None:
    assert Settings().workspace == Path.cwd()


def test_default_web_host_is_loopback_not_all_interfaces() -> None:
    assert Settings().web_host == "127.0.0.1"


def test_default_web_port_is_in_range() -> None:
    assert 1 <= Settings().web_port <= 65535


def test_plan_accepts_an_empty_task_list() -> None:
    plan = Plan(tasks=[], rationale="nothing left to do")
    assert plan.tasks == []
    assert plan.rationale == "nothing left to do"


def test_plan_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValidationError) as excinfo:
        _duplicate_id_plan()
    assert "duplicate task ids" in str(excinfo.value)


def test_plan_rejects_a_dependency_on_an_unknown_id() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Plan(
            tasks=[TaskModel(id="1", description="a", dependencies=["nope"])],
            rationale="dangling",
        )
    assert "nope" in str(excinfo.value)


def test_plan_accepts_a_dependency_on_a_sibling_task() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b", dependencies=["1"]),
        ],
        rationale="ordered",
    )
    assert plan.tasks[1].dependencies == ["1"]


def test_task_model_rejects_a_blank_id() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TaskModel(id="  ", description="nameless")
    assert "blank" in str(excinfo.value)


def test_task_model_rejects_an_unknown_status() -> None:
    with pytest.raises(ValidationError) as excinfo:
        TaskModel(id="1", description="a", status="half-done")
    assert "half-done" in str(excinfo.value)


@pytest.mark.parametrize("status", KNOWN_STATUSES)
def test_task_model_accepts_every_known_status(status: str) -> None:
    assert TaskModel(id="1", description="a", status=status).status == status


def test_task_model_defaults_to_pending_with_no_files_or_dependencies() -> None:
    task = TaskModel(id="1", description="a")
    assert (task.status, task.files, task.dependencies, task.result) == (
        "pending",
        [],
        [],
        None,
    )


def test_test_result_defaults_to_no_failures() -> None:
    assert Verdict(passed=False, summary="nothing").failures == []


def test_get_settings_returns_the_process_wide_singleton() -> None:
    assert get_settings() is config.settings


def test_get_settings_returns_the_same_object_on_repeated_calls() -> None:
    assert get_settings() is get_settings()
