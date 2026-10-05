"""Tests for the domain models.

These pin the behaviours that silently corrupted runs before: duplicate task
ids, list defaults shared between model instances, and a skipped test run that
could be mistaken for a pass.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from furrow.models import Plan, TaskModel, TaskStatus, TestResult, WorkerResult


class _Holder(BaseModel):
    """Throwaway model used to prove TaskStatus survives pydantic validation.

    Round-tripping through a field is the only way to know the enum still behaves
    like a value once it has been serialised, which is what the orchestrator and
    the persistence layer do to it.
    """

    status: TaskStatus


def test_task_status_values() -> None:
    assert TaskStatus.COMPLETED == "completed"
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.SKIPPED.value == "skipped"
    assert {status.value for status in TaskStatus} == {
        "pending",
        "running",
        "completed",
        "failed",
        "skipped",
    }


def test_task_status_round_trips_through_a_field() -> None:
    holder = _Holder(status=TaskStatus.FAILED)
    assert holder.status is TaskStatus.FAILED
    assert holder.model_dump() == {"status": TaskStatus.FAILED}
    assert _Holder.model_validate({"status": "running"}).status is TaskStatus.RUNNING


def test_skipped_is_independent_of_passed() -> None:
    passing = TestResult(passed=True, summary="ok")
    assert passing.passed is True
    # A pass is not automatically "ran", but the common case must not have to
    # carry the extra flag.
    assert passing.skipped is False

    # The defect this field exists for: a run with no test runner used to be
    # reported as the literal summary "No test runner found." and graded as a
    # pass. `skipped` is what lets a caller tell "unknown" from "green", and it
    # is deliberately not derived from `passed`.
    skipped = TestResult(passed=False, summary="no test runner found", skipped=True)
    assert skipped.skipped is True
    assert skipped.passed is False

    # Even an inconsistently filled pair stays representable; nothing coerces
    # one flag into the other.
    assert TestResult(passed=True, summary="", skipped=True).skipped is True


def test_test_result_is_not_collected_by_pytest() -> None:
    # pytest collects any class whose name starts with "Test", so importing this
    # one into a test module used to raise a PytestCollectionWarning. Asserting
    # the attribute is the honest, non-brittle version of asserting on warnings.
    assert TestResult.__test__ is False


def test_plan_rejects_duplicate_task_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate task id"):
        Plan(
            tasks=[
                TaskModel(id="1", description="first"),
                TaskModel(id="2", description="second"),
                TaskModel(id="1", description="first again"),
            ],
            rationale="why",
        )


def test_empty_plan_is_valid() -> None:
    plan = Plan()
    assert plan.tasks == []
    assert plan.rationale == ""


def test_task_model_defaults() -> None:
    task = TaskModel(id="1", description="do the thing")
    assert task.status is TaskStatus.PENDING
    assert task.attempts == 0
    assert task.files == []
    assert task.dependencies == []
    assert task.result is None


def test_task_model_list_defaults_are_not_shared() -> None:
    # The classic mutable-default bug: with `= []` on the class the second
    # instance would inherit everything appended to the first.
    first = TaskModel(id="1", description="a")
    first.files.append("a.py")
    first.dependencies.append("0")

    second = TaskModel(id="2", description="b")
    assert second.files == []
    assert second.dependencies == []

    plan = Plan()
    plan.tasks.append(first)
    assert Plan().tasks == []


def test_blank_file_and_dependency_entries_are_rejected() -> None:
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="a", files=["   "])
    with pytest.raises(ValidationError):
        TaskModel(id="1", description="a", dependencies=[""])

    task = TaskModel(id="1", description="a", files=["  a.py  "])
    assert task.files == ["a.py"]


def test_list_defaults_are_not_shared_on_result_models() -> None:
    worker = WorkerResult()
    worker.written_files.append("a.py")
    result = TestResult(passed=True, summary="ok")
    result.failures.append("boom")

    assert WorkerResult().written_files == []
    assert TestResult(passed=True, summary="ok").failures == []


def test_worker_result_defaults() -> None:
    result = WorkerResult()
    assert result.summary == ""
    assert result.written_files == []
    assert result.issue is None
