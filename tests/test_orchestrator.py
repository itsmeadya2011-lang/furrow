"""Tests for the Orchestrator class."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import MAX_RETRIES, Orchestrator


@pytest.fixture
def mock_settings() -> MagicMock:
    s = MagicMock()
    s.max_parallel_tasks = 5
    s.max_cycles = 0
    return s


@pytest.fixture
def mock_client(mock_settings: MagicMock) -> MagicMock:
    c = MagicMock()
    c.settings = mock_settings
    return c


@pytest.fixture
def sample_plan() -> Plan:
    return Plan(
        tasks=[
            TaskModel(id="1", description="task one"),
            TaskModel(id="2", description="task two"),
        ],
        rationale="test plan",
    )


# ---------------------------------------------------------------------------
# _get_tasks / _is_done
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_tasks_empty_before_cycle(mock_client: MagicMock) -> None:
    orch = Orchestrator(goal="test", client=mock_client)
    assert orch._get_tasks() == []


@pytest.mark.asyncio
async def test_get_tasks_returns_plan_tasks(mock_client: MagicMock, sample_plan: Plan) -> None:
    orch = Orchestrator(goal="test", client=mock_client)
    orch.current_plan = sample_plan
    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[1].id == "2"


@pytest.mark.asyncio
async def test_is_done_all_completed(mock_client: MagicMock, sample_plan: Plan) -> None:
    for t in sample_plan.tasks:
        t.status = "completed"
    orch = Orchestrator(goal="test", client=mock_client)
    orch.current_plan = sample_plan
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_has_failures(mock_client: MagicMock, sample_plan: Plan) -> None:
    sample_plan.tasks[0].status = "failed"
    orch = Orchestrator(goal="test", client=mock_client)
    orch.current_plan = sample_plan
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_partial(mock_client: MagicMock, sample_plan: Plan) -> None:
    sample_plan.tasks[0].status = "completed"
    # task 2 still "pending"
    orch = Orchestrator(goal="test", client=mock_client)
    orch.current_plan = sample_plan
    assert orch._is_done() is False


# ---------------------------------------------------------------------------
# max_cycles
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_cycles_enforced(mock_client: MagicMock) -> None:
    """Loop stops after max_cycles even when _is_done returns False."""
    mock_client.settings.max_cycles = 2
    orch = Orchestrator(goal="test", client=mock_client)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    orch._is_done = MagicMock(return_value=False)
    await asyncio.wait_for(orch.run(), timeout=15)
    assert orch.cycles == 2


# ---------------------------------------------------------------------------
# output_callback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_callback_receives_messages(mock_client: MagicMock) -> None:
    captured: list[str] = []
    orch = Orchestrator(goal="test", client=mock_client, output_callback=captured.append)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    orch._is_done = MagicMock(return_value=True)
    await asyncio.wait_for(orch.run(), timeout=15)
    assert len(captured) > 0


# ---------------------------------------------------------------------------
# retry logic in _run_worker
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_worker_retries_then_succeeds(mock_client: MagicMock) -> None:
    orch = Orchestrator(goal="test", client=mock_client)
    task = TaskModel(id="1", description="worker task")

    mock_worker_instance = MagicMock()
    mock_worker_instance.run = AsyncMock(side_effect=[RuntimeError("transient"), "success"])

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        MockWorker.return_value = mock_worker_instance
        result = await orch._run_worker(task)

    assert result == "success"
    assert mock_worker_instance.run.call_count == 2


@pytest.mark.asyncio
async def test_run_worker_retries_all_fail(mock_client: MagicMock) -> None:
    orch = Orchestrator(goal="test", client=mock_client)
    task = TaskModel(id="1", description="worker task")

    mock_worker_instance = MagicMock()
    mock_worker_instance.run = AsyncMock(side_effect=RuntimeError("always fails"))

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        MockWorker.return_value = mock_worker_instance
        with pytest.raises(RuntimeError, match="always fails"):
            await orch._run_worker(task)

    assert mock_worker_instance.run.call_count == MAX_RETRIES


# ---------------------------------------------------------------------------
# semaphore / max_parallel_tasks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_semaphore_limited_concurrency(mock_client: MagicMock) -> None:
    """No more than max_parallel_tasks workers should run concurrently."""
    mock_client.settings.max_cycles = 1
    mock_client.settings.max_parallel_tasks = 2

    current = 0
    peak = 0

    async def counting_run(*args, **kwargs) -> str:
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return "ok"

    mock_worker_instance = MagicMock()
    mock_worker_instance.run = counting_run

    plan = Plan(
        tasks=[TaskModel(id=str(i), description=f"task {i}") for i in range(5)],
        rationale="concurrency test",
    )

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value = mock_worker_instance
        MockTester.return_value.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="ok", failures=[])
        )

        # Create Orchestrator inside the patch block so PlannerAgent is mocked
        orch = Orchestrator(goal="test", client=mock_client)
        await asyncio.wait_for(orch.run(), timeout=30)

    assert peak <= 2
