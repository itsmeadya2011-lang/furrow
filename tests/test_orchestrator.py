from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def _orchestrator(settings: Settings, **kwargs: Any) -> Orchestrator:
    return Orchestrator(goal="do something", client=LLMClient(settings=settings), **kwargs)


def test_init(mock_settings: Settings) -> None:
    orch = _orchestrator(mock_settings)
    assert orch.goal == "do something"
    assert orch.cycles == 0
    assert orch.tasks == []


def test_is_done_empty_tasks(mock_settings: Settings) -> None:
    orch = _orchestrator(mock_settings)
    assert orch._is_done() is True


def test_is_done_all_completed(mock_settings: Settings) -> None:
    orch = _orchestrator(mock_settings)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_any_failed(mock_settings: Settings) -> None:
    orch = _orchestrator(mock_settings)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_any_pending(mock_settings: Settings) -> None:
    orch = _orchestrator(mock_settings)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


async def test_max_cycles_enforced(mock_settings: Settings) -> None:
    mock_settings.max_cycles = 2
    orch = _orchestrator(mock_settings)
    orch._cycle = AsyncMock()  # type: ignore[assignment]
    # Keep the loop running until max_cycles is reached.
    orch._is_done = MagicMock(return_value=False)  # type: ignore[assignment]
    await orch.run()
    assert orch.cycles == 2
    assert orch._cycle.await_count == 2


async def test_max_parallel_tasks(mock_settings: Settings) -> None:
    mock_settings.max_parallel_tasks = 3
    orch = _orchestrator(mock_settings)

    plan = Plan(
        tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
        rationale="test",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    captured: dict[str, Any] = {}

    real_semaphore = asyncio.Semaphore

    class FakeSemaphore(real_semaphore):
        def __init__(self, value: int) -> None:
            captured["value"] = value
            super().__init__(value)

    with patch("furrow.core.orchestrator.asyncio.Semaphore", FakeSemaphore):
        with patch("furrow.core.orchestrator.WorkerAgent") as WorkerAgentMock:
            WorkerAgentMock.return_value.run = AsyncMock(return_value="ok")
            with patch("furrow.core.orchestrator.TesterAgent") as TesterAgentMock:
                TesterAgentMock.return_value.run = AsyncMock(
                    return_value=MagicMock(passed=True, summary="ok", failures=[])
                )
                await orch._cycle()

    assert captured["value"] == 3


async def test_output_callback(mock_settings: Settings) -> None:
    messages: list[str] = []
    orch = Orchestrator(
        goal="x",
        client=LLMClient(settings=mock_settings),
        output_callback=messages.append,
    )
    orch._print("hello")
    assert messages == ["hello"]


async def test_cycle_stores_tasks(mock_settings: Settings) -> None:
    orch = _orchestrator(mock_settings)

    plan = Plan(
        tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
        rationale="test",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as WorkerAgentMock:
        WorkerAgentMock.return_value.run = AsyncMock(return_value="ok")
        with patch("furrow.core.orchestrator.TesterAgent") as TesterAgentMock:
            TesterAgentMock.return_value.run = AsyncMock(
                return_value=MagicMock(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()

    assert len(orch.tasks) == 2
    assert all(t.status == "completed" for t in orch.tasks)


async def test_retry_failed_tasks(mock_settings: Settings) -> None:
    orch = _orchestrator(mock_settings)

    plan = Plan(
        tasks=[TaskModel(id="1", description="a")],
        rationale="test",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    with patch("furrow.core.orchestrator.WorkerAgent") as WorkerAgentMock:
        # First run raises, second run succeeds.
        WorkerAgentMock.return_value.run = AsyncMock(
            side_effect=[RuntimeError("boom"), "fixed"]
        )
        with patch("furrow.core.orchestrator.TesterAgent") as TesterAgentMock:
            TesterAgentMock.return_value.run = AsyncMock(
                return_value=MagicMock(passed=True, summary="ok", failures=[])
            )
            await orch._cycle()

    assert orch.tasks[0].status == "completed"
    assert orch.tasks[0].result == "fixed"