from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_get_tasks_empty_when_no_plan():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator._current_plan = None
    assert orchestrator._get_tasks() == []


def test_get_tasks_returns_from_current_plan():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    orchestrator = Orchestrator(goal="test", client=client)
    plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="r")
    orchestrator._current_plan = plan
    assert orchestrator._get_tasks() == plan.tasks


def test_is_done_true_when_all_completed():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    orchestrator = Orchestrator(goal="test", client=client)
    plan = Plan(tasks=[TaskModel(id="1", description="t", status="completed")], rationale="r")
    orchestrator._current_plan = plan
    assert orchestrator._is_done() is True


def test_is_done_false_when_task_failed():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    orchestrator = Orchestrator(goal="test", client=client)
    plan = Plan(tasks=[TaskModel(id="1", description="t", status="failed")], rationale="r")
    orchestrator._current_plan = plan
    assert orchestrator._is_done() is False


def test_is_done_true_when_max_cycles_reached():
    settings = Settings(max_cycles=2)
    client = MagicMock()
    client.settings = settings
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.cycles = 2
    assert orchestrator._is_done() is True


def test_effective_goal_no_fixes():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    orchestrator = Orchestrator(goal="build auth", client=client)
    assert orchestrator._effective_goal() == "build auth"


def test_effective_goal_with_fixes():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    orchestrator = Orchestrator(goal="build auth", client=client)
    orchestrator._fixes = ["test_a failed", "test_b failed"]
    expected = "Fix failing tests:\n- test_a failed\n- test_b failed\n\nOriginal goal: build auth"
    assert orchestrator._effective_goal() == expected


@pytest.mark.asyncio
async def test_cycle_handles_planning_failure():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    client.list_files.return_value = []
    orchestrator = Orchestrator(goal="test", client=client, on_event=lambda msg: None)
    orchestrator.planner = MagicMock()
    orchestrator.planner.plan = AsyncMock(side_effect=RuntimeError("plan boom"))
    await orchestrator._cycle()
    assert orchestrator._current_plan is None


@pytest.mark.asyncio
async def test_cycle_success_with_mocked_agents():
    settings = Settings()
    client = MagicMock()
    client.settings = settings
    client.list_files.return_value = []
    orchestrator = Orchestrator(goal="test", client=client, on_event=lambda msg: None)

    plan = Plan(tasks=[TaskModel(id="1", description="t", status="pending")], rationale="r")
    orchestrator.planner = MagicMock()
    orchestrator.planner.plan = AsyncMock(return_value=plan)

    mock_worker = AsyncMock(return_value="done")
    mock_tester = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockWorker.return_value.run = mock_worker
        MockTester.return_value.run = mock_tester
        await orchestrator._cycle()

    assert orchestrator._current_plan == plan
    assert plan.tasks[0].status == "completed"
    mock_worker.assert_called_once()
    mock_tester.assert_called_once()