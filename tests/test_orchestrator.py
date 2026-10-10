from unittest.mock import AsyncMock, Mock

import pytest

from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import Orchestrator


def test_orchestrator_init():
    mock_client = Mock()
    mock_client.settings.max_cycles = 5
    orchestrator = Orchestrator(goal="Test goal", client=mock_client)
    assert orchestrator.goal == "Test goal"
    assert orchestrator.client == mock_client
    assert orchestrator.cycles == 0
    assert orchestrator.tasks == []
    assert orchestrator._stopped is False


def test_is_done_no_tasks():
    mock_client = Mock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    assert orchestrator._is_done() is True


def test_is_done_all_completed():
    mock_client = Mock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    orchestrator.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="completed"),
    ]
    assert orchestrator._is_done() is True


def test_is_done_has_pending():
    mock_client = Mock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    orchestrator.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_has_failed():
    mock_client = Mock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    orchestrator.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_max_cycles():
    mock_client = Mock()
    mock_client.settings.max_cycles = 3
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    orchestrator.cycles = 3
    assert orchestrator._is_done() is True


def test_merge_tasks_new():
    mock_client = Mock()
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    new_tasks = [
        TaskModel(id="1", description="task1", status="pending"),
        TaskModel(id="2", description="task2", status="pending"),
    ]
    orchestrator._merge_tasks(new_tasks)
    assert len(orchestrator.tasks) == 2
    assert orchestrator.tasks[0].id == "1"
    assert orchestrator.tasks[1].id == "2"


def test_merge_tasks_update_existing():
    mock_client = Mock()
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    orchestrator.tasks = [TaskModel(id="1", description="task1", status="pending")]
    new_tasks = [TaskModel(id="1", description="task1", status="completed", result="done")]
    orchestrator._merge_tasks(new_tasks)
    assert len(orchestrator.tasks) == 1
    assert orchestrator.tasks[0].status == "completed"
    assert orchestrator.tasks[0].result == "done"


def test_compact_context():
    mock_client = Mock()
    orchestrator = Orchestrator(goal="Test goal", client=mock_client)
    orchestrator.tasks = [
        TaskModel(id="1", description="task1", status="completed", result="done"),
        TaskModel(id="2", description="task2", status="pending"),
    ]
    ctx = orchestrator._compact_context()
    assert ctx["goal"] == "Test goal"
    assert ctx["cycles"] == 0
    assert ctx["completed"] == 1
    assert ctx["failed"] == 0
    assert ctx["pending"] == 1
    assert len(ctx["task_summary"]) == 2


def test_stop():
    mock_client = Mock()
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    assert orchestrator._stopped is False
    orchestrator.stop()
    assert orchestrator._stopped is True


def test_get_tasks():
    mock_client = Mock()
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    assert orchestrator._get_tasks() == []
    orchestrator.tasks = [TaskModel(id="1", description="task1")]
    assert len(orchestrator._get_tasks()) == 1


@pytest.mark.asyncio
async def test_cycle_no_pending_tasks():
    """When planner returns tasks but all are already completed, cycle should return early."""
    mock_client = Mock()
    mock_client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    orchestrator.tasks = [TaskModel(id="1", description="task1", status="completed")]

    mock_plan = Plan(tasks=[], rationale="done")
    orchestrator.planner = Mock()
    orchestrator.planner.plan = AsyncMock(return_value=mock_plan)

    # Should not raise
    await orchestrator._cycle()