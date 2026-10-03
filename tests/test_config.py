"""Unit tests for ``furrow.config``: env loading and the pydantic models."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel
from furrow.config import TestResult as _TestResult  # aliased: pytest collects Test* names
from furrow.config import settings as default_settings


@pytest.fixture(autouse=True)
def _clean_furrow_env(monkeypatch):
    """No ambient ``FURROW_*`` variable may reach a default-value assertion."""
    for key in [k for k in os.environ if k.upper().startswith("FURROW_")]:
        monkeypatch.delenv(key, raising=False)


# --------------------------------------------------------------------------- #
# Environment loading
# --------------------------------------------------------------------------- #


def test_env_prefix_makes_furrow_variables_load(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    # pydantic-settings reads the environment when the model is built.
    assert Settings(_env_file=None).max_parallel_tasks == 9


def test_unprefixed_variables_are_ignored(monkeypatch):
    monkeypatch.setenv("MAX_PARALLEL_TASKS", "9")
    assert Settings(_env_file=None).max_parallel_tasks == 5


def test_several_prefixed_variables_load(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_CYCLES", "4")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "DEBUG")
    loaded = Settings(_env_file=None)
    assert loaded.max_cycles == 4
    assert loaded.log_level == "DEBUG"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("anthropic", Provider.ANTHROPIC),
        ("openai", Provider.OPENAI),
        ("ollama", Provider.OLLAMA),
    ],
)
def test_provider_parses_from_the_environment(monkeypatch, raw, expected):
    monkeypatch.setenv("FURROW_PROVIDER", raw)
    assert Settings(_env_file=None).provider is expected


def test_unknown_provider_is_rejected(monkeypatch):
    # ValueError covers both pydantic's ValidationError and pydantic-settings'
    # SettingsError, which is what the version in use raises for a bad env value.
    monkeypatch.setenv("FURROW_PROVIDER", "not-a-provider")
    with pytest.raises(ValueError):
        Settings(_env_file=None)


def test_explicit_init_arguments_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    assert Settings(_env_file=None, max_parallel_tasks=2).max_parallel_tasks == 2


def test_model_copy_updates_without_mutating_the_original(tmp_path):
    original = Settings(_env_file=None, workspace=tmp_path)
    updated = original.model_copy(update={"max_cycles": 3, "log_level": "WARNING"})
    assert updated.max_cycles == 3
    assert updated.log_level == "WARNING"
    assert original.max_cycles == 0
    assert original.log_level == "INFO"
    assert updated.workspace == tmp_path


# --------------------------------------------------------------------------- #
# Field defaults
# --------------------------------------------------------------------------- #


def test_settings_field_defaults(tmp_path):
    loaded = Settings(_env_file=None, workspace=tmp_path)
    assert loaded.provider is Provider.ANTHROPIC
    assert loaded.model == "claude-sonnet-4-20250514"
    assert loaded.planner_model == "claude-3-5-haiku-20241022"
    assert loaded.worker_model == "claude-3-5-sonnet-20241022"
    assert loaded.tester_model == "claude-3-5-sonnet-20241022"
    assert loaded.anthropic_api_key is None
    assert loaded.openai_api_key is None
    assert loaded.ollama_base_url == "http://localhost:11434"
    assert loaded.max_parallel_tasks == 5
    assert loaded.max_cycles == 0
    assert loaded.log_level == "INFO"
    assert loaded.workspace == tmp_path


def test_workspace_defaults_to_the_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Settings(_env_file=None).workspace == Path.cwd()


def test_module_singleton_is_a_settings_instance():
    assert isinstance(default_settings, Settings)
    assert isinstance(default_settings.workspace, Path)
    assert isinstance(default_settings.provider, Provider)


def test_provider_is_a_string_enum():
    assert Provider.OLLAMA == "ollama"
    assert Provider.OLLAMA.value == "ollama"
    assert [p.value for p in Provider] == ["anthropic", "openai", "ollama"]


# --------------------------------------------------------------------------- #
# TaskModel / Plan / TestResult
# --------------------------------------------------------------------------- #


def test_task_model_defaults():
    task = TaskModel(id="1", description="do it")
    assert task.id == "1"
    assert task.description == "do it"
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_keeps_explicit_values():
    task = TaskModel(
        id="a",
        description="d",
        files=["a.py"],
        dependencies=["b"],
        status="done",
        result="wrote it",
    )
    assert task.files == ["a.py"]
    assert task.dependencies == ["b"]
    assert task.status == "done"
    assert task.result == "wrote it"


def test_task_model_default_lists_are_not_shared_between_instances():
    first = TaskModel(id="1", description="a")
    second = TaskModel(id="2", description="b")
    first.files.append("mutated.py")
    first.dependencies.append("ghost")
    assert second.files == []
    assert second.dependencies == []


def test_plan_holds_tasks_and_rationale():
    plan = Plan(tasks=[TaskModel(id="1", description="d")], rationale="because")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "1"
    assert plan.rationale == "because"


def test_plan_accepts_an_empty_task_list():
    plan = Plan(tasks=[], rationale="")
    assert plan.tasks == []
    assert plan.rationale == ""


def test_test_result_defaults():
    result = _TestResult(passed=True, summary="all good")
    assert result.passed is True
    assert result.summary == "all good"
    assert result.failures == []


def test_test_result_failure_lists_are_not_shared():
    first = _TestResult(passed=False, summary="a")
    second = _TestResult(passed=False, summary="b")
    first.failures.append("boom")
    assert second.failures == []
