from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_is_done_no_tasks():
    orch = Orchestrator(goal="do something")
    orch._tasks = []
    orch._last_test_result = None
    assert orch._is_done() is True


def test_is_done_all_completed_passing():
    orch = Orchestrator(goal="do something")
    orch._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


def test_is_done_with_failures():
    orch = Orchestrator(goal="do something")
    orch._tasks = [
        TaskModel(id="1", description="task 1", status="failed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is False


def test_is_done_tests_failed():
    orch = Orchestrator(goal="do something")
    orch._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch._last_test_result = TestResult(passed=False, summary="broken", failures=["x"])
    assert orch._is_done() is False


def test_get_tasks_returns_stored():
    orch = Orchestrator(goal="do something")
    stored = [TaskModel(id="1", description="task 1")]
    orch._tasks = stored
    assert orch._get_tasks() is stored


@pytest.mark.asyncio
async def test_max_cycles_enforced():
    orch = Orchestrator(goal="do something")
    cycle_calls = []

    async def fake_cycle():
        cycle_calls.append(orch.cycles)

    orch._cycle = fake_cycle
    orch._is_done = lambda: False
    orch.client = MagicMock()
    orch.client.settings = MagicMock()
    orch.client.settings.log_level = "INFO"
    orch.client.settings.max_cycles = 1

    with patch.object(Orchestrator, "_print", new=AsyncMock()):
        await orch.run()

    assert len(cycle_calls) == 1
    assert orch.cycles == 1
