from __future__ import annotations

import os
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def _make_plan(tasks=None, rationale="test plan"):
    """Helper to build a Plan with tasks."""
    if tasks is None:
        tasks = [TaskModel(id="1", description="task 1", status="pending")]
    return Plan(tasks=tasks, rationale=rationale)


@pytest.fixture
def mock_client(tmp_path):
    """An LLMClient mock with temp workspace settings."""
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(workspace=tmp_path)
    return client


# --- _is_done / _get_tasks ---

def test_is_done_no_tasks():
    """_is_done returns False when no tasks exist (loop should continue)."""
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    orch = Orchestrator(goal="test", client=client)
    assert orch._is_done() is False


def test_is_done_all_completed():
    """_is_done returns True when all tasks are completed."""
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    orch = Orchestrator(goal="test", client=client)
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_with_failures():
    """_is_done returns False when any task has failed."""
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    orch = Orchestrator(goal="test", client=client)
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_pending_tasks():
    """_is_done returns False when some tasks are still pending."""
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    orch = Orchestrator(goal="test", client=client)
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    assert orch._is_done() is False


def test_get_tasks_returns_stored():
    """_get_tasks returns the stored _tasks list."""
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    orch = Orchestrator(goal="test", client=client)
    test_tasks = [TaskModel(id="1", description="t", status="completed")]
    orch._tasks = test_tasks
    assert orch._get_tasks() is test_tasks


# --- Cycle execution ---

async def test_cycle_stores_tasks(tmp_path):
    """After _cycle(), plan tasks are stored in _tasks."""
    settings = Settings(workspace=tmp_path)
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    plan = _make_plan(tasks=[
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ])

    orch = Orchestrator(goal="test", client=client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        await orch._cycle()

    assert len(orch._tasks) == 2
    assert all(t.status == "completed" for t in orch._tasks)


async def test_cycle_marks_failed_worker():
    """A worker exception marks the task as failed."""
    settings = Settings(workspace=tmp_path)
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    plan = _make_plan(tasks=[TaskModel(id="1", description="task 1")])

    orch = Orchestrator(goal="test", client=client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(side_effect=Exception("worker error"))
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        await orch._cycle()

    assert orch._tasks[0].status == "failed"
    assert "worker error" in orch._tasks[0].result


async def test_cycle_test_failure_rewrites_goal(tmp_path):
    """When tests fail, the goal is rewritten for the next cycle."""
    settings = Settings(workspace=tmp_path)
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    plan = _make_plan()

    orch = Orchestrator(goal="Build feature X", client=client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["err1", "err2"])
        )
        await orch._cycle()

    assert "Fix failing tests" in orch.goal
    assert "err1" in orch.goal
    assert "err2" in orch.goal


async def test_cycle_empty_plan(tmp_path):
    """A cycle with no planned tasks is a no-op for task storage."""
    settings = Settings(workspace=tmp_path)
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    plan = Plan(tasks=[], rationale="nothing to do")

    orch = Orchestrator(goal="test", client=client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        await orch._cycle()

    assert len(orch._tasks) == 0
    MockWorker.assert_not_called()
    MockTester.assert_not_called()


# --- max_cycles ---

async def test_max_cycles_limits_runs(tmp_path):
    """run() stops after max_cycles even if goal is not complete."""
    settings = Settings(max_cycles=3, workspace=tmp_path)
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    # Empty plan → no tasks → _is_done returns False → relies on max_cycles
    plan = Plan(tasks=[], rationale="nothing")

    orch = Orchestrator(goal="test", client=client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent"), \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["x"])
        )
        await orch.run()

    assert orch.cycles == 3


async def test_max_cycles_zero_is_unlimited(tmp_path):
    """max_cycles=0 means unlimited; loop runs until _is_done."""
    settings = Settings(max_cycles=0, workspace=tmp_path)
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    plan = _make_plan(tasks=[TaskModel(id="1", status="completed")])

    orch = Orchestrator(goal="test", client=client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        await orch.run()

    assert orch.cycles == 1  # All tasks completed → _is_done → stop


# --- max_parallel_tasks semaphore ---

async def test_max_parallel_tasks_respected(tmp_path):
    """max_parallel_tasks setting is passed to the semaphore."""
    settings = Settings(
        workspace=tmp_path,
        max_parallel_tasks=2,
        max_cycles=1,
    )
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    plan = _make_plan(tasks=[
        TaskModel(id="1", description="t1"),
        TaskModel(id="2", description="t2"),
        TaskModel(id="3", description="t3"),
        TaskModel(id="4", description="t4"),
    ])

    orch = Orchestrator(goal="test", client=client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["x"])
        )

        with patch("furrow.core.orchestrator.asyncio.Semaphore") as MockSem:
            MockWorker.return_value.run = AsyncMock(return_value="done")
            await orch._cycle()

    assert MockSem.called
    assert MockSem.call_args[0][0] == 2


# --- Workspace handling ---

async def test_orchestrator_uses_workspace(tmp_path, monkeypatch):
    """Orchestrator changes to the configured workspace directory."""
    original_cwd = Path.cwd()
    monkeypatch.chdir(original_cwd)  # Ensures cwd is restored after test
    settings = Settings(workspace=tmp_path)
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    orch = Orchestrator(goal="test", client=client)
    assert Path.cwd() == tmp_path


# --- Test failure → rewrite goal → retry ---

async def test_test_failure_triggers_retry(tmp_path):
    """Test failures rewrite the goal, enabling a retry cycle."""
    settings = Settings(workspace=tmp_path)
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    plan = _make_plan(tasks=[TaskModel(id="1", status="completed")])

    orch = Orchestrator(goal="Add feature", client=client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="broken", failures=["TypeError in main.py"])
        )
        await orch._cycle()

    assert "Fix failing tests" in orch.goal
    assert "TypeError" in orch.goal
