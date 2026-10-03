"""Tests for :mod:`furrow.config`.

Covers the data models exchanged between the orchestrator and the agents plus
``Settings`` env-var loading. The models are the shared vocabulary of the whole
system, so their defaults and the string/enum coercion rules matter: several
modules assign plain strings (``task.status = "failed"``) and compare against
plain strings (``task.status == "completed"``).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import (
    CycleReport,
    Plan,
    Provider,
    RunResult,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
)


@pytest.fixture
def isolated_settings_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Make ``Settings()`` see only defaults.

    Runs in an empty directory so no stray ``.env`` file (which ``Settings``
    reads) can leak in, and clears any inherited ``FURROW_*`` variables.
    """
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("FURROW_"):
            monkeypatch.delenv(key)


def test_task_model_defaults() -> None:
    task = TaskModel(id="t1", description="do the thing")
    assert task.status is TaskStatus.PENDING
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_task_model_requires_id_and_description() -> None:
    with pytest.raises(ValidationError):
        TaskModel()  # type: ignore[call-arg]


def test_task_model_default_lists_are_not_shared_between_instances() -> None:
    # default_factory, not a shared mutable default.
    first = TaskModel(id="a", description="a")
    second = TaskModel(id="b", description="b")
    first.files.append("x")
    assert second.files == []
    assert first.dependencies is not second.dependencies


def test_assignment_coerces_plain_string_status_to_enum() -> None:
    # Regression: the orchestrator assigns plain strings. With validate_assignment
    # these must land as TaskStatus members, not raw strings.
    task = TaskModel(id="t1", description="d")
    task.status = "failed"
    assert task.status is TaskStatus.FAILED


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("pending", TaskStatus.PENDING),
        ("running", TaskStatus.RUNNING),
        ("completed", TaskStatus.COMPLETED),
        ("failed", TaskStatus.FAILED),
        ("skipped", TaskStatus.SKIPPED),
    ],
)
def test_every_status_string_assignment_is_coerced(raw: str, expected: TaskStatus) -> None:
    task = TaskModel(id="t1", description="d")
    task.status = raw
    assert task.status is expected


def test_status_compares_equal_to_plain_string() -> None:
    task = TaskModel(id="t1", description="d", status="completed")
    assert task.status == "completed"
    assert task.status != "failed"
    assert isinstance(task.status, str)


def test_invalid_status_assignment_raises_validation_error() -> None:
    task = TaskModel(id="t1", description="d")
    with pytest.raises(ValidationError):
        task.status = "not-a-status"
    # The failed assignment left the previous value untouched.
    assert task.status is TaskStatus.PENDING


def test_task_status_is_a_str_enum_matching_its_value() -> None:
    assert issubclass(TaskStatus, str)
    assert [s.value for s in TaskStatus] == ["pending", "running", "completed", "failed", "skipped"]
    assert TaskStatus.FAILED == "failed"
    assert TaskStatus("running") is TaskStatus.RUNNING


def test_provider_is_a_str_enum_matching_its_value() -> None:
    assert issubclass(Provider, str)
    assert [p.value for p in Provider] == ["anthropic", "openai", "ollama"]
    assert Provider.OPENAI == "openai"
    assert Provider("ollama") is Provider.OLLAMA


def test_plan_requires_tasks_and_rationale() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=[])  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        Plan(rationale="why")  # type: ignore[call-arg]


def test_plan_holds_tasks_and_rationale() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="d")], rationale="small")
    assert plan.rationale == "small"
    assert plan.tasks[0].id == "1"


def test_test_result_defaults_failures_to_empty() -> None:
    result = TestResult(passed=True, summary="ok")
    assert result.failures == []
    assert result.passed is True


def test_test_result_sets_test_flag_false_so_pytest_skips_it() -> None:
    # pytest collects any imported name starting with "Test"; without this flag the
    # BaseModel import emits a PytestCollectionWarning. Do not remove the flag.
    assert TestResult.__test__ is False
    assert "failures" in TestResult.model_fields


def test_cycle_report_defaults_everything_but_cycle() -> None:
    report = CycleReport(cycle=2)
    assert report.cycle == 2
    assert report.planned == 0
    assert report.completed == 0
    assert report.failed == 0
    assert report.tests_passed is None
    assert report.summary == ""


def test_cycle_report_requires_cycle() -> None:
    with pytest.raises(ValidationError):
        CycleReport()  # type: ignore[call-arg]


def test_run_result_defaults() -> None:
    result = RunResult(goal="ship it")
    assert result.goal == "ship it"
    assert result.cycles == 0
    assert result.stopped_reason == "goal_complete"
    assert result.tasks == []
    assert result.test_results == []
    assert result.reports == []


def test_run_result_requires_goal() -> None:
    with pytest.raises(ValidationError):
        RunResult()  # type: ignore[call-arg]


def test_run_result_default_lists_are_not_shared_between_instances() -> None:
    first = RunResult(goal="a")
    second = RunResult(goal="b")
    first.reports.append(CycleReport(cycle=1))
    assert second.reports == []


def test_settings_loads_prefixed_env_var(
    monkeypatch: pytest.MonkeyPatch, isolated_settings_env: None
) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    assert Settings().max_parallel_tasks == 9


def test_settings_ignores_unprefixed_env_var(
    monkeypatch: pytest.MonkeyPatch, isolated_settings_env: None
) -> None:
    # Proves the FURROW_ prefix is load-bearing, not decorative.
    monkeypatch.setenv("MAX_PARALLEL_TASKS", "42")
    assert Settings().max_parallel_tasks == 5


def test_settings_parses_enum_and_optional_str_from_env(
    monkeypatch: pytest.MonkeyPatch, isolated_settings_env: None
) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "openai")
    monkeypatch.setenv("FURROW_TEST_COMMAND", "pytest -q")
    monkeypatch.setenv("FURROW_ANTHROPIC_API_KEY", "sk-test")
    settings = Settings()
    assert settings.provider is Provider.OPENAI
    assert settings.test_command == "pytest -q"
    assert settings.anthropic_api_key == "sk-test"


def test_settings_defaults_without_any_environment(isolated_settings_env: None) -> None:
    settings = Settings()
    assert settings.provider is Provider.ANTHROPIC
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.max_task_retries == 1
    assert settings.test_timeout == 120
    assert settings.test_command is None
    assert settings.log_level == "INFO"
    assert settings.workspace == Path.cwd()


def test_settings_accepts_explicit_arguments(isolated_settings_env: None) -> None:
    settings = Settings(max_parallel_tasks=2, provider=Provider.OLLAMA)
    assert settings.max_parallel_tasks == 2
    assert settings.provider is Provider.OLLAMA
