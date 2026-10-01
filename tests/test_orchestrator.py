"""Tests for the orchestrator's cycle tracking and completion logic."""
from __future__ import annotations

import pytest

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def _make_settings(max_cycles: int = 0) -> Settings:
    return Settings(max_cycles=max_cycles, provider="anthropic", model="claude-test")


@pytest.mark.asyncio
async def test_is_done_true_when_all_tasks_completed_and_tests_passed():
    """When all tasks are completed and tests passed, _is_done returns True."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_false_when_tests_failed():
    """When tests failed, _is_done returns False even if all tasks completed."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [TaskModel(id="1", description="task 1", status="completed")]
    orch._tests_passed = False
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_false_when_task_failed():
    """When a task failed, _is_done returns False."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_false_when_tasks_incomplete():
    """When some tasks haven't run yet, _is_done returns False."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    orch._tests_passed = True
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_true_when_no_tasks():
    """When there are no tasks and tests passed, _is_done returns True."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.tasks = []
    orch._tests_passed = True
    assert orch._is_done() is True
