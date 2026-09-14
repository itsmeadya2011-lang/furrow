"""Tests for Furrow core functionality."""
from __future__ import annotations

import pytest

from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_dependencies():
    """Test that tasks can have dependencies."""
    t1 = TaskModel(id="1", description="first", dependencies=[])
    t2 = TaskModel(id="2", description="second", dependencies=["1"])
    assert t2.dependencies == ["1"]


def test_task_model_defaults():
    """Test that TaskModel has sensible defaults."""
    t = TaskModel(id="1", description="test")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_empty():
    """Test that an empty plan is valid."""
    p = Plan(tasks=[], rationale="nothing to do")
    assert len(p.tasks) == 0


def test_test_result_with_failures():
    """Test TestResult with failures."""
    t = TestResult(passed=False, summary="broken", failures=["err1", "err2"])
    assert t.passed is False
    assert len(t.failures) == 2