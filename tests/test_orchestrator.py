from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_get_tasks_empty():
    orch = Orchestrator(goal="test")
    assert orch.tasks == []


def test_is_done_no_tasks():
    orch = Orchestrator(goal="test")
    assert orch._is_done() is True


def test_is_done_all_completed_tests_pass():
    orch = Orchestrator(goal="test")
    orch.tasks = [TaskModel(id="1", description="task", status="completed")]
    orch._tests_passed = True
    assert orch._is_done() is True


def test_is_done_all_completed_tests_fail():
    orch = Orchestrator(goal="test")
    orch.tasks = [TaskModel(id="1", description="task", status="completed")]
    orch._tests_passed = False
    assert orch._is_done() is False


def test_is_done_has_failed():
    orch = Orchestrator(goal="test")
    orch.tasks = [TaskModel(id="1", description="task", status="failed")]
    orch._tests_passed = False
    assert orch._is_done() is False


def test_is_done_partial():
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="pending"),
    ]
    orch._tests_passed = False
    assert orch._is_done() is False


def test_strip_ansi():
    orch = Orchestrator(goal="test")
    text = "\x1b[31mRed\x1b[0m\x1b[1;32mGreen\x1b[0m"
    result = orch._strip_ansi(text)
    assert result == "RedGreen"
    assert "\x1b" not in result


@pytest.mark.asyncio
async def test_max_cycles_respected():
    orch = Orchestrator(goal="test")
    orch.max_cycles = 2
    with (
        patch.object(orch, "_cycle", new_callable=AsyncMock) as mock_cycle,
        patch.object(orch, "_is_done", return_value=False),
    ):
        await orch.run()
        assert mock_cycle.await_count == 2


@pytest.mark.asyncio
async def test_log_callback_receives_messages():
    received: list[str] = []

    async def callback(msg: str) -> None:
        received.append(msg)

    orch = Orchestrator(goal="test", log_callback=callback)
    orch.max_cycles = 1
    fake_plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=fake_plan)

    with (
        patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as mock_tester_cls,
    ):
        mock_worker = MagicMock()
        mock_worker.run = AsyncMock(return_value="done")
        mock_worker_cls.return_value = mock_worker

        mock_tester = MagicMock()
        mock_tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok"))
        mock_tester_cls.return_value = mock_tester

        await orch.run()
        await asyncio.sleep(0.1)

    assert len(received) > 0
    assert any("Cycle 1" in msg for msg in received)
