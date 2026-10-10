from unittest.mock import AsyncMock, Mock, patch

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


def test_merge_tasks():
    mock_client = Mock()
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    initial = TaskModel(id="1", description="initial", status="pending")
    orchestrator.tasks = [initial]

    new_tasks = [
        TaskModel(id="1", description="updated", status="completed", result="done"),
        TaskModel(id="2", description="new", status="pending"),
    ]
    orchestrator._merge_tasks(new_tasks)
    assert len(orchestrator.tasks) == 2
    assert orchestrator.tasks[0].status == "completed"
    assert orchestrator.tasks[0].result == "done"
    assert orchestrator.tasks[1].id == "2"


def test_stop_method():
    mock_client = Mock()
    orchestrator = Orchestrator(goal="Test", client=mock_client)
    assert orchestrator._stopped is False
    orchestrator.stop()
    assert orchestrator._stopped is True


def test_compact_context():
    mock_client = Mock()
    orchestrator = Orchestrator(goal="Test goal", client=mock_client)
    orchestrator.cycles = 2
    orchestrator.tasks = [
        TaskModel(id="1", description="do work", status="completed", result="done"),
        TaskModel(id="2", description="more work", status="pending"),
        TaskModel(id="3", description="broken", status="failed", result="error"),
    ]
    context = orchestrator._compact_context()
    assert context["goal"] == "Test goal"
    assert context["cycles"] == 2
    assert len(context["task_summary"]) == 3
    assert context["completed"] == 1
    assert context["pending"] == 1
    assert context["failed"] == 1
