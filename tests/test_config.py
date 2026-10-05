"""Tests for the configuration models in :mod:`furrow.config`.

These are the shapes the agents serialise and deserialise at their boundaries, so
the defaults matter: a missing field default would surface as a plan that silently
loses its rationale rather than as a parse error.

The :class:`Settings` cases are the other half.  ``Settings`` is populated from
``FURROW_``-prefixed environment variables, and the module-level singleton is built
at import time, so every test here constructs its own instance and the defaults test
neutralises the environment first.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult

# Environment variables that would silently redefine the value under test.
_DEFAULT_SOURCES = (
    "FURROW_MAX_CYCLES",
    "FURROW_MAX_PARALLEL_TASKS",
    "FURROW_MAX_CONSECUTIVE_FAILURES",
    "FURROW_LLM_MAX_RETRIES",
    "FURROW_HOST",
    "FURROW_PORT",
)


@pytest.fixture
def no_config_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove any ambient ``FURROW_`` variable that would mask a field default."""
    for name in _DEFAULT_SOURCES:
        monkeypatch.delenv(name, raising=False)


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="d")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == "pending"
    assert task.result is None


def test_task_model_keeps_supplied_fields() -> None:
    task = TaskModel(
        id="2",
        description="wire the widget",
        files=["furrow/widget.py"],
        dependencies=["1"],
        status="done",
        result="wrote it",
    )
    assert task.files == ["furrow/widget.py"]
    assert task.dependencies == ["1"]
    assert task.status == "done"
    assert task.result == "wrote it"


def test_task_model_list_defaults_are_not_shared() -> None:
    first = TaskModel(id="1", description="d")
    second = TaskModel(id="2", description="d")
    first.files.append("leaked.py")
    first.dependencies.append("0")
    # A shared mutable default would leak one task's files into every other task.
    assert second.files == []
    assert second.dependencies == []


def test_test_result_defaults() -> None:
    result = TestResult(passed=True, summary="s")
    assert result.passed is True
    assert result.summary == "s"
    assert result.failures == []


def test_test_result_keeps_failures() -> None:
    result = TestResult(passed=False, summary="2 failed", failures=["tests/test_a.py::test_x"])
    assert result.passed is False
    assert result.failures == ["tests/test_a.py::test_x"]


def test_plan_accepts_a_minimal_set_of_tasks() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="small")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "small"


def test_plan_requires_tasks() -> None:
    with pytest.raises(ValidationError):
        Plan(rationale="no tasks given")


def test_plan_requires_rationale() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="do thing")])


def test_settings_reads_prefixed_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "2")
    config = Settings()
    assert config.max_cycles == 7
    assert config.provider == Provider.OPENAI
    assert config.max_parallel_tasks == 2


def test_settings_defaults(no_config_env: None) -> None:
    # _env_file=None drops the .env source so a developer's local file cannot
    # redefine what "the default" means.
    config = Settings(_env_file=None)
    assert config.max_cycles == 0
    assert config.max_parallel_tasks == 5
    assert config.max_consecutive_failures == 3
    assert config.llm_max_retries == 3
    assert config.host == "127.0.0.1"
    assert config.port == 8000


def test_settings_workspace_comes_from_the_constructor(tmp_path: Path) -> None:
    assert Settings(workspace=tmp_path).workspace == tmp_path
