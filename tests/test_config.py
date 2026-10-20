"""Tests for the configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from furrow.config import Plan, Provider, Settings, TaskModel, TaskStatus, TestResult


def test_task_status_is_a_string_enum() -> None:
    assert issubclass(TaskStatus, str)
    assert TaskStatus.PENDING == "pending"
    assert TaskStatus("completed") is TaskStatus.COMPLETED
    assert TaskStatus.COMPLETED.value == "completed"
    assert {status.value for status in TaskStatus} == {
        "pending",
        "in_progress",
        "completed",
        "failed",
    }


def test_provider_is_a_string_enum() -> None:
    assert issubclass(Provider, str)
    assert Provider.ANTHROPIC == "anthropic"
    assert {provider.value for provider in Provider} == {"anthropic", "openai", "ollama"}


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="do the thing")
    assert task.files == []
    assert task.dependencies == []
    assert task.status == TaskStatus.PENDING
    assert task.result is None
    assert task.files_changed == []


def test_task_model_accepts_explicit_values() -> None:
    task = TaskModel(
        id="7",
        description="write the parser",
        files=["src/parser.py"],
        dependencies=["1"],
        status=TaskStatus.COMPLETED,
        result="done",
        files_changed=["src/parser.py"],
    )
    assert task.files == ["src/parser.py"]
    assert task.dependencies == ["1"]
    assert task.status is TaskStatus.COMPLETED
    assert task.result == "done"
    assert task.files_changed == ["src/parser.py"]


def test_test_result_is_not_collected_by_pytest() -> None:
    assert TestResult.__test__ is False


def test_test_result_defaults() -> None:
    result = TestResult(passed=True, summary="ok")
    assert result.passed is True
    assert result.failures == []


def test_plan_requires_tasks_and_rationale() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="why")
    assert plan.rationale == "why"
    assert plan.tasks[0].id == "1"


def test_plan_rejects_a_missing_rationale() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="x")])  # type: ignore[call-arg]


def test_plan_rejects_tasks_that_are_not_a_list() -> None:
    with pytest.raises(ValidationError):
        Plan(tasks="not-a-list", rationale="why")  # type: ignore[arg-type]


def test_settings_defaults() -> None:
    fields = Settings.model_fields
    assert fields["provider"].default is Provider.ANTHROPIC
    assert fields["max_parallel_tasks"].default == 5
    assert fields["max_cycles"].default == 0
    assert fields["max_consecutive_failures"].default == 3
    assert fields["apply_worker_edits"].default is True
    assert fields["anthropic_api_key"].default is None
    assert fields["openai_api_key"].default is None
