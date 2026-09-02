from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.fixture
def mock_client():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    return client


def make_task(id: str, status: str = "completed", result: str = "done") -> TaskModel:
    return TaskModel(id=id, description=f"task {id}", status=status, result=result)


def test_get_tasks_returns_plan_tasks(mock_client):
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(
        tasks=[make_task("1"), make_task("2")],
        rationale="test plan",
    )
    tasks = orchestrator._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"


def test_get_tasks_empty_when_no_plan(mock_client):
    orchestrator = Orchestrator(goal="test", client=mock_client)
    assert orchestrator._get_tasks() == []


def test_is_done_completed(mock_client):
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(
        tasks=[make_task("1", "completed"), make_task("2", "completed")],
        rationale="test plan",
    )
    assert orchestrator._is_done() is True


def test_is_done_pending(mock_client):
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(
        tasks=[make_task("1", "pending"), make_task("2", "pending")],
        rationale="test plan",
    )
    assert orchestrator._is_done() is False


def test_is_done_failed(mock_client):
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(
        tasks=[make_task("1", "completed"), make_task("2", "failed")],
        rationale="test plan",
    )
    assert orchestrator._is_done() is False


def test_is_done_no_tasks(mock_client):
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(tasks=[], rationale="test plan")
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles(mock_client):
    settings = Settings(max_cycles=2)
    mock_client.settings = settings

    planner = MagicMock(spec=PlannerAgent)
    planner.plan = AsyncMock(
        return_value=Plan(tasks=[make_task("1")], rationale="test")
    )

    worker = MagicMock(spec=WorkerAgent)
    worker.run = AsyncMock(return_value="done")

    tester = MagicMock(spec=TesterAgent)
    tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

    orchestrator = Orchestrator(goal="test", client=mock_client, settings=settings)
    orchestrator.planner = planner
    orchestrator._cycle = AsyncMock()

    is_done_calls = 0

    def fake_is_done() -> bool:
        nonlocal is_done_calls
        is_done_calls += 1
        return is_done_calls > 1

    orchestrator._is_done = fake_is_done

    await orchestrator.run()

    assert orchestrator.cycles == 2


@pytest.mark.asyncio
async def test_progress_callback_invoked(mock_client):
    messages: list[str] = []

    async def callback(msg: str) -> None:
        messages.append(msg)

    orchestrator = Orchestrator(goal="test", client=mock_client, progress_callback=callback)
    orchestrator.current_plan = Plan(tasks=[make_task("1", "completed")], rationale="test plan")

    await orchestrator._emit("hello")

    assert messages == ["hello"]
