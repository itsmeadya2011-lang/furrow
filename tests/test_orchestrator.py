import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def make_mock_client():
    """Create a mock LLMClient that returns a minimal valid plan."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = 3
    client.settings.workspace = MagicMock()
    client.settings.workspace.__truediv__ = lambda self, other: f"{self}/{other}"
    client.complete = AsyncMock()
    client.read_file = AsyncMock()
    client.write_file = AsyncMock()

    # Default: return a plan with one task
    plan = Plan(
        tasks=[TaskModel(id="1", description="dummy task", files=[])],
        rationale="test plan",
    )
    client.complete.return_value = '{"tasks": [{"id": "1", "description": "dummy"}], "rationale": "test"}'
    return client


@pytest.fixture
def mock_client():
    return make_mock_client()


@pytest.mark.asyncio
async def test_is_done_empty_tasks():
    """_is_done returns True when there are no tasks."""
    orch = Orchestrator(goal="test")
    orch.tasks = []
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_all_completed():
    """_is_done returns True when all tasks are completed."""
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    for t in orch.tasks:
        t.status = "completed"
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_has_failed():
    """_is_done returns False when there are failed tasks."""
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_partial_completion():
    """_is_done returns False when not all tasks are completed."""
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_get_tasks_returns_stored_tasks():
    """_get_tasks returns the tasks stored on the orchestrator."""
    orch = Orchestrator(goal="test")
    tasks = [TaskModel(id="1", description="task 1")]
    orch.tasks = tasks
    assert orch._get_tasks() == tasks


@pytest.mark.asyncio
async def test_max_cycles_enforced():
    """Orchestrator stops after max_cycles even if goal isn't complete."""
    orch = Orchestrator(goal="test")
    orch.client = make_mock_client()
    orch.client.settings.max_cycles = 2
    orch.max_cycles = 2

    # Patch the cycle methods so we don't need real LLM calls
    orch._cycle = AsyncMock()
    orch._is_done = MagicMock(return_value=False)

    await orch.run()

    assert orch.cycles == 2
    assert orch._cycle.call_count == 2


@pytest.mark.asyncio
async def test_max_cycles_zero_is_infinite_until_done():
    """When max_cycles is 0, the loop runs until _is_done returns True."""
    orch = Orchestrator(goal="test")
    orch.client = make_mock_client()
    orch.client.settings.max_cycles = 0
    orch.max_cycles = 0

    call_count = 0

    async def fake_cycle():
        nonlocal call_count
        call_count += 1
        orch.tasks = [TaskModel(id="1", description="task", status="completed")]

    orch._cycle = fake_cycle

    await orch.run()

    assert call_count == 1  # _is_done returns True after first cycle


@pytest.mark.asyncio
async def test_cycle_updates_task_status():
    """_cycle stores tasks and updates their status based on results."""
    from furrow.config import Plan, TaskModel

    orch = Orchestrator(goal="test")
    orch.client = make_mock_client()

    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="test",
    )
    test_result = TestResult(passed=True, summary="all passed", failures=[])

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)):
        with patch.object(WorkerAgent, "run", new=AsyncMock(return_value="implemented")):
            with patch.object(TesterAgent, "run", new=AsyncMock(return_value=test_result)):
                await orch._cycle()

    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"
    assert orch.tasks[0].result == "implemented"


@pytest.mark.asyncio
async def test_task_failure_sets_status():
    """_cycle marks tasks as failed when WorkerAgent raises."""
    from furrow.config import Plan, TaskModel

    orch = Orchestrator(goal="test")
    orch.client = make_mock_client()

    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="test",
    )
    test_result = TestResult(passed=False, summary="failed", failures=["error"])

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)):
        with patch.object(WorkerAgent, "run", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with patch.object(TesterAgent, "run", new=AsyncMock(return_value=test_result)):
                await orch._cycle()

    assert orch.tasks[0].status == "failed"
    assert "boom" in orch.tasks[0].result


@pytest.mark.asyncio
async def test_semaphore_limits_parallelism():
    """Tasks are executed with max_parallel_tasks semaphore."""
    from furrow.config import Plan, TaskModel

    orch = Orchestrator(goal="test")
    orch.client = make_mock_client()
    orch.client.settings.max_parallel_tasks = 2

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
            TaskModel(id="3", description="task 3"),
        ],
        rationale="test",
    )
    test_result = TestResult(passed=True, summary="ok", failures=[])

    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)):
        with patch.object(WorkerAgent, "run", new=AsyncMock(return_value="done")):
            with patch.object(TesterAgent, "run", new=AsyncMock(return_value=test_result)):
                await orch._cycle()

    assert all(t.status == "completed" for t in orch.tasks)
