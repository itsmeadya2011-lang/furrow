"""Smoke tests for the pydantic models in :mod:`furrow.config`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from furrow.config import Plan, TaskModel, TaskStatus
from furrow.config import TestResult as Verdict


def test_plan_parse():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert plan.tasks[0].description == "do thing"
    assert plan.rationale == "ok"


def test_test_result():
    result = Verdict(passed=True, summary="ok", failures=[])
    assert result.passed is True
    assert result.failures == []


def test_task_status_is_a_str_enum():
    assert TaskStatus.COMPLETED == "completed"
    assert isinstance(TaskStatus.PENDING, str)
    assert TaskStatus("failed") is TaskStatus.FAILED
    assert TaskStatus.SKIPPED.value == "skipped"


def test_task_model_coerces_a_status_string_into_the_enum():
    task = TaskModel(id="1", description="do thing", status="completed")

    assert task.status is TaskStatus.COMPLETED

    task.status = "failed"

    assert task.status is TaskStatus.FAILED
    assert task.status == "failed"
    with pytest.raises(ValidationError):
        task.status = "not-a-status"