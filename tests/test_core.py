from __future__ import annotations

import pytest

from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert len(p.tasks) == 0


def test_test_result_with_failures():
    t = TestResult(
        passed=False,
        summary="2 tests failed",
        failures=["test_a::test_x failed", "test_b::test_y failed"],
    )
    assert t.passed is False
    assert len(t.failures) == 2


def test_task_model_dependencies():
    t = TaskModel(
        id="2",
        description="implement feature b",
        files=["src/b.py"],
        dependencies=["1"],
    )
    assert t.dependencies == ["1"]
    assert t.files == ["src/b.py"]