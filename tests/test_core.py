from __future__ import annotations

import pytest

from furrow.config import FileEdit, Plan, Settings, TaskModel, TaskResult, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_file_edit_defaults():
    e = FileEdit(file="x.py", operation="create")
    assert e.old is None
    assert e.new is None
    assert e.content is None


def test_task_result_defaults():
    t = TaskResult(summary="s")
    assert t.edits == []
    assert t.issues == []


def test_task_model_defaults():
    t = TaskModel(id="1", description="d")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None


def test_plan_requires_tasks():
    with pytest.raises(ValueError):
        Plan(rationale="no tasks")


def test_settings_model_field():
    s = Settings()
    assert isinstance(s.model, str)
