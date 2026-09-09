import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_get_tasks_returns_plan_tasks():
    mock_client = MagicMock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="test",
    )
    assert len(orchestrator._get_tasks()) == 1
    assert orchestrator._get_tasks()[0].id == "1"


def test_orchestrator_is_done_no_tasks():
    mock_client = MagicMock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(tasks=[], rationale="none")
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_completed():
    mock_client = MagicMock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ],
        rationale="test",
    )
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_has_failed():
    mock_client = MagicMock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ],
        rationale="test",
    )
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_mixed_status():
    mock_client = MagicMock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="pending"),
        ],
        rationale="test",
    )
    assert orchestrator._is_done() is False


def test_orchestrator_max_cycles_enforced():
    mock_client = MagicMock()
    mock_client.settings.max_cycles = 2
    orchestrator = Orchestrator(goal="test", client=mock_client)
    orchestrator.planner = MagicMock()
    orchestrator.planner.plan = AsyncMock(side_effect=ValueError("bad plan"))

    with patch("furrow.core.orchestrator.console"):
        asyncio.run(orchestrator.run())

    assert orchestrator.cycles == 2
