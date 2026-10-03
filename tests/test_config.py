"""Tests for the pydantic data layer in :mod:`furrow.config`.

Two concerns live here: the models must tolerate the partial output an LLM
produces (``Plan`` with no rationale, ``TestResult`` with nothing but a verdict),
while still *rejecting* the typos that used to silently corrupt run state --
``status="complete"`` is a validation error now that ``TaskStatus`` is a real
enum. And ``Settings`` must read ``FURROW_``-prefixed environment variables and
nothing else.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from furrow.config import (
    Plan,
    Provider,
    TaskModel,
    TaskStatus,
    TestResult,
)
from tests.conftest import offline_settings

ALL_STATUSES = list(TaskStatus)


def test_plan_with_no_arguments_is_an_empty_plan() -> None:
    plan = Plan()
    assert plan.tasks == []
    assert plan.rationale == ""


def test_plan_accepts_tasks_without_a_rationale() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")])
    assert plan.rationale == ""


def test_plan_accepts_an_empty_task_list_from_json() -> None:
    plan = Plan.model_validate({"tasks": []})
    assert plan.tasks == []


def test_plan_accepts_an_empty_json_object() -> None:
    plan = Plan.model_validate({})
    assert plan.tasks == []
    assert plan.rationale == ""


def test_test_result_requires_only_the_verdict() -> None:
    result = TestResult(passed=False)
    assert result.passed is False
    assert result.summary == ""
    assert result.failures == []


def test_test_result_rejects_a_payload_without_a_verdict() -> None:
    with pytest.raises(ValidationError):
        TestResult.model_validate({"summary": "no verdict here"})


def test_task_model_rejects_an_unknown_status_typo() -> None:
    with pytest.raises(ValidationError):
        TaskModel.model_validate({"id": "1", "description": "d", "status": "complete"})


def test_task_status_is_a_string_enum() -> None:
    assert isinstance(TaskStatus.COMPLETED, str)
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.COMPLETED.value == "completed"


def test_completed_task_status_matches_a_plain_string() -> None:
    task = TaskModel.model_validate(
        {"id": "1", "description": "d", "status": "completed"}
    )
    assert task.status == "completed"
    assert task.status == TaskStatus.COMPLETED


def test_every_status_has_the_documented_value() -> None:
    assert [status.value for status in ALL_STATUSES] == [
        "pending",
        "running",
        "completed",
        "failed",
        "blocked",
        "skipped",
    ]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TaskStatus.PENDING, False),
        (TaskStatus.RUNNING, False),
        (TaskStatus.COMPLETED, True),
        (TaskStatus.FAILED, True),
        (TaskStatus.BLOCKED, True),
        (TaskStatus.SKIPPED, True),
    ],
)
def test_is_terminal_is_true_for_every_status_the_run_will_not_revisit(
    status: TaskStatus,
    expected: bool,
) -> None:
    task = TaskModel(id="1", description="d", status=status)
    assert task.is_terminal() is expected


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (TaskStatus.PENDING, False),
        (TaskStatus.RUNNING, False),
        (TaskStatus.COMPLETED, True),
        (TaskStatus.FAILED, False),
        (TaskStatus.BLOCKED, False),
        (TaskStatus.SKIPPED, False),
    ],
)
def test_is_done_is_true_only_for_a_successful_task(
    status: TaskStatus,
    expected: bool,
) -> None:
    task = TaskModel(id="1", description="d", status=status)
    assert task.is_done() is expected


def test_task_defaults_to_pending_with_no_files() -> None:
    task = TaskModel(id="1", description="d")
    assert task.status == TaskStatus.PENDING
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None
    assert task.error is None
    assert task.attempts == 0


def test_task_file_lists_are_not_shared_between_instances() -> None:
    first = TaskModel(id="1", description="d")
    second = TaskModel(id="2", description="d")
    first.files.append("a.py")
    assert second.files == []


def test_plan_task_lists_are_not_shared_between_instances() -> None:
    first = Plan()
    first.tasks.append(TaskModel(id="1", description="d"))
    assert len(Plan().tasks) == 0


def test_test_result_failure_lists_are_not_shared_between_instances() -> None:
    first = TestResult(passed=False)
    first.failures.append("boom")
    assert len(TestResult(passed=False).failures) == 0


def test_settings_defaults(clean_env: None, tmp_path: Path) -> None:
    settings = offline_settings(workspace=tmp_path)
    assert settings.provider == Provider.ANTHROPIC
    assert settings.model == "claude-sonnet-4-20250514"
    assert settings.planner_model == "claude-3-5-haiku-20241022"
    assert settings.worker_model == "claude-3-5-sonnet-20241022"
    assert settings.tester_model == "claude-3-5-sonnet-20241022"
    assert settings.anthropic_api_key is None
    assert settings.openai_api_key is None
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.max_tokens == 8192
    assert settings.max_parallel_tasks == 5
    assert settings.max_cycles == 0
    assert settings.max_consecutive_failures == 3
    assert settings.max_task_attempts == 2
    assert settings.test_command is None
    assert settings.test_timeout == 300
    assert settings.state_file is None
    assert settings.workspace == tmp_path
    assert settings.log_level == "INFO"


def test_settings_default_workspace_is_the_current_directory(clean_env: None) -> None:
    assert offline_settings().workspace == Path.cwd()


def test_settings_reads_max_tokens_from_the_environment(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_MAX_TOKENS", "128")
    assert offline_settings().max_tokens == 128


def test_settings_reads_max_cycles_from_the_environment(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_MAX_CYCLES", "7")
    assert offline_settings().max_cycles == 7


def test_settings_reads_max_parallel_tasks_from_the_environment(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "9")
    assert offline_settings().max_parallel_tasks == 9


def test_settings_reads_the_provider_from_the_environment(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "ollama")
    assert offline_settings().provider == Provider.OLLAMA


def test_settings_reads_a_path_setting_from_the_environment(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("FURROW_STATE_FILE", str(tmp_path / "custom.json"))
    assert offline_settings().state_file == tmp_path / "custom.json"


def test_settings_ignores_an_environment_variable_without_the_prefix(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MAX_TOKENS", "128")
    assert offline_settings().max_tokens == 8192


def test_settings_rejects_an_unknown_provider(
    clean_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_PROVIDER", "not-a-provider")
    with pytest.raises(ValidationError):
        offline_settings()


def test_test_result_is_hidden_from_pytest_collection() -> None:
    assert TestResult.__test__ is False


def test_pytest_marker_is_not_a_test_result_field() -> None:
    assert "__test__" not in TestResult.model_fields


def test_pytest_marker_is_absent_from_the_serialised_result() -> None:
    assert "__test__" not in TestResult(passed=True).model_dump()


def test_task_model_json_round_trip() -> None:
    task = TaskModel(
        id="t1",
        description="wire things up",
        files=["a.py", "b.py"],
        dependencies=["t0"],
        status=TaskStatus.FAILED,
        result="partial",
        error="ValueError: nope",
        attempts=2,
    )
    restored = TaskModel.model_validate_json(task.model_dump_json())
    assert restored == task
    assert restored.status == TaskStatus.FAILED


def test_plan_json_round_trip_keeps_task_order_and_statuses() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="t1", description="first", status=TaskStatus.COMPLETED),
            TaskModel(id="t2", description="second", status=TaskStatus.BLOCKED),
            TaskModel(id="t3", description="third"),
        ],
        rationale="because",
    )
    restored = Plan.model_validate_json(plan.model_dump_json())
    assert restored == plan
    assert [task.id for task in restored.tasks] == ["t1", "t2", "t3"]
    assert restored.tasks[1].status == TaskStatus.BLOCKED


def test_test_result_json_round_trip() -> None:
    result = TestResult(passed=False, summary="1 failed", failures=["tests/test_a.py"])
    restored = TestResult.model_validate_json(result.model_dump_json())
    assert restored == result


def test_serialised_status_is_the_plain_string_value() -> None:
    task = TaskModel(id="t1", description="d", status=TaskStatus.COMPLETED)
    assert '"status":"completed"' in task.model_dump_json()
