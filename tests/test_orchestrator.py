"""Tests for the Orchestrator loop logic and task tracking."""
from __future__ import annotations

from unittest.mock import Mock

import pytest
from rich.console import Console

from furrow.config import Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_orchestrator(goal: str = "test goal", **kwargs) -> Orchestrator:
    """Create an Orchestrator with a mock client to avoid API calls."""
    return Orchestrator(goal=goal, client=Mock(), **kwargs)


def test_get_tasks_returns_empty_initially():
    """_get_tasks() should return an empty list when no tasks have been planned."""
    orch = _make_orchestrator()
    assert orch._get_tasks() == []


def test_get_tasks_returns_tracked_tasks():
    """_get_tasks() should return tasks stored in all_tasks, not an empty list."""
    orch = _make_orchestrator()
    task = TaskModel(id="1", description="do thing", status="completed")
    orch.all_tasks = [task]
    assert orch._get_tasks() == [task]


def test_is_done_no_tasks():
    """Orchestrator should be done when there are no tasks."""
    orch = _make_orchestrator()
    orch.all_tasks = []
    orch.test_results = []
    assert orch._is_done() is True


def test_is_done_all_completed_tests_passed():
    """Should be done when all tasks completed and tests passed."""
    orch = _make_orchestrator()
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    orch.test_results = [TestResult(passed=True, summary="ok", failures=[])]
    assert orch._is_done() is True


def test_is_done_tests_failed():
    """Should NOT be done when tests failed (will retry to fix)."""
    orch = _make_orchestrator()
    orch.all_tasks = [TaskModel(id="1", description="t1", status="completed")]
    orch.test_results = [TestResult(passed=False, summary="fail", failures=["err"])]
    assert orch._is_done() is False


def test_is_done_task_failed():
    """Should NOT be done when a task failed."""
    orch = _make_orchestrator()
    orch.all_tasks = [TaskModel(id="1", description="t1", status="failed", result="err")]
    orch.test_results = [TestResult(passed=True, summary="ok", failures=[])]
    assert orch._is_done() is False


def test_is_done_task_pending():
    """Should NOT be done when tasks are still pending."""
    orch = _make_orchestrator()
    orch.all_tasks = [TaskModel(id="1", description="t1", status="pending")]
    orch.test_results = []
    assert orch._is_done() is False


def test_is_done_max_cycles_reached():
    """Should be done when max_cycles is reached, even if tasks are pending."""
    custom_settings = Settings(max_cycles=2)
    orch = _make_orchestrator(settings=custom_settings)
    orch.cycles = 2
    orch.all_tasks = [TaskModel(id="1", description="t1", status="pending")]
    orch.test_results = []
    assert orch._is_done() is True


def test_is_done_max_cycles_not_reached():
    """Should NOT be done by max_cycles if cycles haven't reached the limit."""
    custom_settings = Settings(max_cycles=5)
    orch = _make_orchestrator(settings=custom_settings)
    orch.cycles = 2
    orch.all_tasks = [TaskModel(id="1", description="t1", status="completed")]
    orch.test_results = [TestResult(passed=True, summary="ok", failures=[])]
    assert orch._is_done() is True  # done because tasks completed and tests passed


def test_is_done_max_cycles_unlimited():
    """With max_cycles=0 (unlimited), should rely on task/test logic."""
    custom_settings = Settings(max_cycles=0)
    orch = _make_orchestrator(settings=custom_settings)
    orch.cycles = 100
    orch.all_tasks = [
        TaskModel(id="1", description="t1", status="completed"),
    ]
    orch.test_results = [TestResult(passed=True, summary="ok", failures=[])]
    assert orch._is_done() is True


def test_console_is_injectable():
    """Orchestrator should accept a custom Console instance."""
    mock_console = Console()
    orch = _make_orchestrator(console=mock_console)
    assert orch.console is mock_console


def test_console_defaults_when_none():
    """Orchestrator should use a default Console when none is provided."""
    orch = _make_orchestrator()
    assert orch.console is not None
    assert isinstance(orch.console, Console)


def test_settings_used_in_constructor():
    """Orchestrator should use provided settings."""
    custom_settings = Settings(model="custom-model", max_cycles=3)
    orch = _make_orchestrator(settings=custom_settings)
    assert orch.settings is custom_settings
    assert orch.settings.model == "custom-model"
    assert orch.settings.max_cycles == 3


def test_instance_isolation():
    """Each Orchestrator instance should have its own task/test tracking."""
    orch1 = _make_orchestrator()
    orch2 = _make_orchestrator()
    orch1.all_tasks = [TaskModel(id="1", description="t1", status="completed")]
    assert orch2.all_tasks == []
    assert orch1._get_tasks() != orch2._get_tasks()
