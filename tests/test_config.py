"""Regression tests for :mod:`furrow.config`.

The load-bearing details pinned here:

* ``TaskStatus`` is a ``str`` subclass, so plain-string comparison, assignment,
  serialisation and dumping all keep working.
* ``Settings`` validates its numeric bounds and its log level, normalises the
  level spelling and applies ``FURROW_*`` environment overrides.
* ``configure_logging`` is safe to call repeatedly and rejects a bad level.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import structlog
from pydantic import ValidationError

from furrow import config
from furrow.config import (
    LOG_LEVELS,
    Plan,
    Provider,
    Settings,
    TaskModel,
    TaskStatus,
    TestResult,
    configure_logging,
)

# ------------------------------------------------------------------ TaskStatus


def test_task_status_is_a_str_subclass() -> None:
    assert issubclass(TaskStatus, str)
    assert isinstance(TaskStatus.COMPLETED, str)
    assert TaskStatus.PENDING == "pending"


def test_task_status_stringifies_like_a_plain_string() -> None:
    assert str(TaskStatus.COMPLETED) == "completed"
    assert repr(TaskStatus.COMPLETED) == "'completed'"
    assert f"{TaskStatus.FAILED}" == "failed"


def test_task_model_accepts_a_plain_string_status() -> None:
    task = TaskModel(id="1", description="d", status="completed")
    assert task.status == "completed"
    assert task.status is TaskStatus.COMPLETED


def test_task_model_coerces_assignment_of_a_plain_string() -> None:
    task = TaskModel(id="1", description="d")
    task.status = "failed"
    assert task.status is TaskStatus.FAILED
    assert task.status == "failed"


def test_task_model_assignment_validates() -> None:
    """``validate_assignment=True`` means a bad status cannot sneak in later."""
    task = TaskModel(id="1", description="d")
    with pytest.raises(ValidationError):
        task.status = "bogus"
    assert task.status is TaskStatus.PENDING


def test_task_model_rejects_an_unknown_status() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="d", status="bogus")


def test_task_model_dump_yields_plain_strings() -> None:
    task = TaskModel(id="1", description="d", status="completed")
    dumped = task.model_dump()
    assert dumped["status"] == "completed"
    json_dumped = task.model_dump(mode="json")
    assert type(json_dumped["status"]) is str
    assert json_dumped["status"] == "completed"
    assert '"status":"completed"' in task.model_dump_json()


def test_task_model_accepts_loose_status_spellings() -> None:
    assert TaskModel(id="1", description="d", status="Completed").status is TaskStatus.COMPLETED
    assert TaskModel(id="1", description="d", status="in-progress").status is TaskStatus.IN_PROGRESS
    assert TaskModel(id="1", description="d", status=" in_progress ").status is (
        TaskStatus.IN_PROGRESS
    )


def test_task_status_is_terminal() -> None:
    assert TaskStatus.COMPLETED.is_terminal is True
    assert TaskStatus.FAILED.is_terminal is True
    assert TaskStatus.SKIPPED.is_terminal is True
    assert TaskStatus.PENDING.is_terminal is False
    assert TaskStatus.IN_PROGRESS.is_terminal is False


def test_task_status_covers_the_whole_vocabulary() -> None:
    assert {member.value for member in TaskStatus} == {
        "pending",
        "in_progress",
        "completed",
        "failed",
        "skipped",
    }


def test_task_model_defaults() -> None:
    task = TaskModel(id=" 1 ", description="  d  ")
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None
    assert task.id == "1"
    assert task.description == "d"


def test_task_model_requires_an_id_and_a_description() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="", description="d")
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="")


# ---------------------------------------------------------------------- Plan


def test_plan_constructs_from_tasks() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="d")], rationale="because")
    assert len(plan.tasks) == 1
    assert plan.rationale == "because"
    assert plan.tasks[0].status is TaskStatus.PENDING


def test_plan_accepts_a_plain_dict_task() -> None:
    plan = Plan(tasks=[{"id": "1", "description": "d"}], rationale="r")
    assert isinstance(plan.tasks[0], TaskModel)
    assert plan.tasks[0].dependencies == []


def test_plan_requires_a_rationale() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="d")])


# ----------------------------------------------------------------- TestResult


def test_test_result_constructs_with_defaults() -> None:
    result = TestResult(passed=True, summary="ok")
    assert result.passed is True
    assert result.failures == []
    assert TestResult(passed=False, summary="s", failures=["a", "b"]).failures == ["a", "b"]


def test_test_result_requires_a_summary() -> None:
    with pytest.raises(ValidationError):
        TestResult(passed=True)


# ------------------------------------------------------------------- Settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.provider is Provider.ANTHROPIC
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.log_level == "INFO"
    assert settings.retry_attempts == 3
    assert settings.llm_max_tokens == 4096
    assert settings.llm_temperature == 0.0
    assert settings.test_timeout == 120


@pytest.mark.parametrize("value", [0, -1, -10])
def test_settings_rejects_non_positive_max_parallel_tasks(value: int) -> None:
    with pytest.raises(ValidationError):
        Settings(max_parallel_tasks=value)


def test_settings_accepts_max_parallel_tasks_of_one() -> None:
    assert Settings(max_parallel_tasks=1).max_parallel_tasks == 1


def test_settings_rejects_negative_max_cycles() -> None:
    with pytest.raises(ValidationError):
        Settings(max_cycles=-1)


def test_settings_accepts_zero_max_cycles_as_unlimited() -> None:
    assert Settings(max_cycles=0).max_cycles == 0


@pytest.mark.parametrize("value", ["nope", "trace", "10", "  "])
def test_settings_rejects_an_unknown_log_level(value: str) -> None:
    with pytest.raises(ValidationError):
        Settings(log_level=value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(" debug ", "DEBUG"), ("warning", "WARNING"), ("Info", "INFO"), ("error", "ERROR")],
)
def test_settings_normalises_the_log_level(value: str, expected: str) -> None:
    assert Settings(log_level=value).log_level == expected


@pytest.mark.parametrize(
    "field_name", ["model", "planner_model", "worker_model", "tester_model", "ollama_base_url"]
)
def test_settings_rejects_whitespace_only_model_names(field_name: str) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: "   "})


@pytest.mark.parametrize("field_name", ["model", "planner_model", "worker_model", "tester_model"])
def test_settings_strips_surrounding_whitespace_from_model_names(field_name: str) -> None:
    assert Settings(**{field_name: "  spaced  "}).model_dump()[field_name] == "spaced"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [("retry_attempts", 0), ("test_timeout", 0), ("retry_min_wait", -1.0), ("llm_max_tokens", 0)],
)
def test_settings_rejects_out_of_range_bounds(field_name: str, value: object) -> None:
    with pytest.raises(ValidationError):
        Settings(**{field_name: value})


def test_settings_parses_the_provider_from_a_string() -> None:
    assert Settings(provider="openai").provider is Provider.OPENAI
    assert Settings(provider="ollama").provider is Provider.OLLAMA
    assert Settings(provider="anthropic").provider == "anthropic"


def test_settings_rejects_an_unknown_provider() -> None:
    with pytest.raises(ValidationError):
        Settings(provider="bedrock")


def test_settings_applies_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MODEL", "env-model")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "4")
    monkeypatch.setenv("FURROW_LOG_LEVEL", "warning")
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    monkeypatch.setenv("FURROW_OLLAMA_BASE_URL", "http://ollama.test:1234")
    settings = Settings()
    assert settings.model == "env-model"
    assert settings.max_parallel_tasks == 9
    assert settings.max_cycles == 4
    assert settings.log_level == "WARNING"
    assert settings.provider is Provider.OLLAMA
    assert settings.ollama_base_url == "http://ollama.test:1234"


def test_environment_overrides_are_reverted_by_monkeypatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MODEL", "env-model")
    assert Settings().model == "env-model"
    monkeypatch.delenv("FURROW_MODEL")
    assert Settings().model != "env-model"


def test_settings_reads_a_dot_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("FURROW_MODEL=from-dot-env\nFURROW_MAX_CYCLES=2\n")
    settings = Settings()
    assert settings.model == "from-dot-env"
    assert settings.max_cycles == 2


def test_explicit_arguments_win_over_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MODEL", "env-model")
    assert Settings(model="explicit").model == "explicit"


def test_settings_does_not_mutate_the_global_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MODEL", "env-model")
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    Settings()
    assert config.settings.model != "env-model"
    assert config.settings.max_cycles != 7


def test_settings_workspace_is_a_path(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path)
    assert isinstance(settings.workspace, Path)
    assert settings.workspace == tmp_path


def test_settings_model_copy_does_not_revalidate_or_mutate(tmp_path: Path) -> None:
    original = Settings(workspace=tmp_path, max_cycles=3)
    clone = original.model_copy(update={"max_cycles": 5})
    assert clone.max_cycles == 5
    assert original.max_cycles == 3


# ---------------------------------------------------------- configure_logging


def test_configure_logging_sets_the_root_level() -> None:
    configure_logging("WARNING")
    assert logging.getLogger().level == logging.WARNING
    configure_logging("DEBUG")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_normalises_the_level_argument() -> None:
    configure_logging(" debug ")
    assert logging.getLogger().level == logging.DEBUG


def test_configure_logging_defaults_to_the_settings_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an argument the level comes from the module level settings."""
    monkeypatch.setattr(config, "settings", Settings(log_level="critical"))
    configure_logging()
    assert logging.getLogger().level == logging.CRITICAL


def test_configure_logging_rejects_a_bad_level() -> None:
    with pytest.raises(ValueError):
        configure_logging("verbose")


def test_configure_logging_rejects_a_bad_settings_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_LOG_LEVEL", "verbose")
    with pytest.raises(ValueError):
        Settings()


def test_configure_logging_is_idempotent() -> None:
    """Repeated calls must not accumulate handlers or processors."""
    configure_logging("INFO")
    handlers_after_first = list(logging.getLogger().handlers)
    processors_after_first = [type(item) for item in structlog.get_config()["processors"]]
    for _ in range(4):
        configure_logging("INFO")
    assert logging.getLogger().handlers == handlers_after_first
    assert [type(item) for item in structlog.get_config()["processors"]] == processors_after_first


def test_log_levels_are_the_documented_set() -> None:
    assert LOG_LEVELS == ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    for level in LOG_LEVELS:
        configure_logging(level)
        assert logging.getLogger().level == getattr(logging, level)
