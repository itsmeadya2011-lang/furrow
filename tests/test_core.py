import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="test")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_orchestrator_initial_state():
    orch = Orchestrator(goal="build a castle")
    assert orch.goal == "build a castle"
    assert orch.cycles == 0
    assert orch.tasks == []
    assert orch._last_test_result is None
    assert orch._on_event is None


def test_orchestrator_with_event_callback():
    events = []

    async def callback(event_type: str, data: dict):
        events.append((event_type, data))

    orch = Orchestrator(goal="test", on_event=callback)
    assert orch._on_event is callback


def test_is_done_max_cycles():
    orch = Orchestrator(goal="test")
    orch.cycles = 5
    orch.client.settings.max_cycles = 3
    assert orch._is_done() is True


def test_is_done_infinite_cycles():
    orch = Orchestrator(goal="test")
    orch.cycles = 5
    orch.client.settings.max_cycles = 0
    # No tasks and no test result -> should be done
    assert orch._is_done() is True


def test_is_done_with_failed_tests():
    orch = Orchestrator(goal="test")
    orch.cycles = 1
    orch._last_test_result = TestResult(
        passed=False, summary="broken", failures=["test_x failed"]
    )
    orch.tasks = [TaskModel(id="1", description="fix", status="completed")]
    # Tests failed -> should NOT be done (continue to fix)
    assert orch._is_done() is False


def test_is_done_with_passed_tests_all_completed():
    orch = Orchestrator(goal="test")
    orch.cycles = 1
    orch._last_test_result = TestResult(
        passed=True, summary="all good", failures=[]
    )
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="completed"),
    ]
    # Tests passed + all tasks completed -> done
    assert orch._is_done() is True


def test_is_done_with_pending_tasks():
    orch = Orchestrator(goal="test")
    orch.cycles = 1
    orch._last_test_result = TestResult(
        passed=True, summary="ok", failures=[]
    )
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="pending"),
    ]
    # Pending tasks exist -> not done
    assert orch._is_done() is False


def test_is_done_with_failed_tasks():
    orch = Orchestrator(goal="test")
    orch.cycles = 1
    orch._last_test_result = TestResult(
        passed=True, summary="ok", failures=[]
    )
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="failed"),
    ]
    # Failed task -> not done (continue to retry)
    assert orch._is_done() is False


def test_get_tasks_returns_stored_tasks():
    orch = Orchestrator(goal="test")
    task1 = TaskModel(id="1", description="task1")
    task2 = TaskModel(id="2", description="task2")
    orch.tasks = [task1, task2]
    assert orch._get_tasks() == [task1, task2]


def test_get_tasks_empty():
    orch = Orchestrator(goal="test")
    assert orch._get_tasks() == []


@pytest.mark.asyncio
async def test_orchestrator_emit_calls_callback():
    events = []

    async def callback(event_type: str, data: dict):
        events.append((event_type, data))

    orch = Orchestrator(goal="test", on_event=callback)
    await orch._emit("test_event", {"key": "value"})
    assert ("test_event", {"key": "value"}) in events


@pytest.mark.asyncio
async def test_orchestrator_emit_without_callback():
    orch = Orchestrator(goal="test")
    # Should not raise even without callback
    await orch._emit("test_event", {"key": "value"})


@pytest.mark.asyncio
async def test_run_worker_with_semaphore():
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="worker result text")
    mock_client.settings = MagicMock()
    mock_client.settings.worker_model = "test-model"

    orch = Orchestrator(goal="test", client=mock_client)
    orch._semaphore = asyncio.Semaphore(2)
    task = TaskModel(id="1", description="do something")

    result = await orch._run_worker(task)
    assert isinstance(result, str)
    assert result == "worker result text"
    assert isinstance(orch._semaphore, asyncio.Semaphore)
