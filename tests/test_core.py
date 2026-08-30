import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_empty():
    o = Orchestrator(goal="test")
    assert o._is_done() is True


def test_orchestrator_is_done_with_completed_tasks():
    o = Orchestrator(goal="test")
    o._latest_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert o._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    o = Orchestrator(goal="test")
    o._latest_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    assert o._is_done() is False


def test_orchestrator_get_tasks():
    o = Orchestrator(goal="test")
    assert o._get_tasks() == []
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    o._latest_plan = plan
    assert o._get_tasks() == plan.tasks


@pytest.mark.asyncio
async def test_orchestrator_run_stops_on_max_cycles():
    mock_client = AsyncMock(spec=LLMClient)
    mock_client.settings.max_cycles = 1
    o = Orchestrator(goal="test", client=mock_client)
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner:
        mock_planner = AsyncMock()
        mock_planner.plan.return_value = Plan(tasks=[], rationale="done")
        MockPlanner.return_value = mock_planner
        await o.run()
        assert o.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_emits_events():
    events: list[str] = []

    async def capture(msg: str) -> None:
        events.append(msg)

    mock_client = AsyncMock(spec=LLMClient)
    mock_client.settings.max_cycles = 0
    o = Orchestrator(goal="test", client=mock_client, on_event=capture)

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner:
        mock_planner = AsyncMock()
        mock_planner.plan.return_value = Plan(tasks=[], rationale="done")
        MockPlanner.return_value = mock_planner
        await o.run()

    assert any("Goal: test" in e for e in events)
    assert any("Planning..." in e for e in events)
