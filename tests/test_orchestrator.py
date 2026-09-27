from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_is_done_no_tasks_returns_false():
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client)
    orch._tasks = []
    orch._last_test_result = None
    assert orch._is_done() is False


def test_is_done_all_completed_tests_passed():
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client)
    orch._tasks = [TaskModel(id="1", description="task1", status="completed")]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


def test_is_done_tasks_incomplete_returns_false():
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client)
    orch._tasks = [TaskModel(id="1", description="task1", status="completed")]
    orch._last_test_result = TestResult(passed=False, summary="fail")
    assert orch._is_done() is False


def test_is_done_some_failed_returns_false():
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client)
    orch._tasks = [TaskModel(id="1", description="task1", status="failed")]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is False


async def test_max_cycles_enforced():
    custom_settings = Settings(max_cycles=1)
    client = MagicMock()
    orch = Orchestrator(goal="test", client=client, settings=custom_settings)

    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="task1")], rationale="test"
        )
    )

    mock_worker = MagicMock()
    mock_worker.run = AsyncMock(return_value="done")
    mock_tester = MagicMock()
    mock_tester.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok")
    )

    with patch(
        "furrow.core.orchestrator.WorkerAgent", return_value=mock_worker
    ), patch(
        "furrow.core.orchestrator.TesterAgent", return_value=mock_tester
    ):
        await orch.run()

    assert orch.cycles == 1
