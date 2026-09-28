"""Tests for Furrow config models and core data structures."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_plan_empty_tasks():
    p = Plan(tasks=[], rationale="nothing to do")
    assert len(p.tasks) == 0
    assert p.rationale == "nothing to do"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_with_failures():
    t = TestResult(passed=False, summary="2 tests failed", failures=["test_a", "test_b"])
    assert t.passed is False
    assert len(t.failures) == 2


def test_task_model_defaults():
    t = TaskModel(id="1", description="implement feature")
    assert t.id == "1"
    assert t.description == "implement feature"
    assert t.files == []  # default empty list
    assert t.dependencies == []  # default empty list
    assert t.status == "pending"  # default status
    assert t.result is None  # default None


def test_task_model_with_dependencies():
    t = TaskModel(
        id="2",
        description="task b",
        dependencies=["1"],
        status="pending",
    )
    assert t.dependencies == ["1"]


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0  # 0 means unlimited
    assert s.log_level == "INFO"
    assert isinstance(s.workspace, Path)


def test_settings_env_override():
    """Settings should respect FURROW_ env prefix."""
    with patch.dict(
        "os.environ",
        {"FURROW_MAX_CYCLES": "3", "FURROW_MAX_PARALLEL_TASKS": "2"},
        clear=True,
    ):
        s = Settings()
        assert s.max_cycles == 3
        assert s.max_parallel_tasks == 2
