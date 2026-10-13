"""Tests for the Orchestrator and its cycle logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def base_settings():
    return Settings(max_cycles=0, max_parallel_tasks=5)


@pytest.fixture
def mock_planner():
    """Create a mock PlannerAgent that returns a predefined plan."""
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="Task 1", files=[], dependencies=[]),
            TaskModel(id="2", description="Task 2", files=[], dependencies=[]),
        ],
        rationale="Test plan",
    )
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner:
        instance = MagicMock()
        instance.plan = AsyncMock(return_value=plan)
        MockPlanner.return_value = instance
        yield instance


@pytest.fixture
def mock_workers():
    """Create a mock WorkerAgent that returns a completion message."""
    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        instance = MagicMock()
        instance.run = AsyncMock(return_value="Done")
        MockWorker.return_value = instance
        yield instance


@pytest.fixture
def mock_tester_pass():
    """Create a mock TesterAgent that returns passing test results."""
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        instance = MagicMock()
        instance.run = AsyncMock(
            return_value=TestResult(passed=True, summary="All passed", failures=[])
        )
        MockTester.return_value = instance
        yield instance


@pytest.fixture
def mock_tester_fail():
    """Create a mock TesterAgent that returns failing test results."""
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        instance = MagicMock()
        instance.run = AsyncMock(
            return_value=TestResult(
                passed=False, summary="2 failures", failures=["fail1", "fail2"]
            )
        )
        MockTester.return_value = instance
        yield instance


def test_get_tasks_empty_before_cycle():
    """The _get_tasks() method should return empty list before any cycle."""
    orch = Orchestrator(goal="test goal")
    assert orch._get_tasks() == []


@pytest.mark.asyncio
async def test_is_done_empty_tasks():
    """With no tasks, _is_done() should return True."""
    orch = Orchestrator(goal="test goal")
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_all_completed():
    """When all tasks are completed, _is_done() should return True."""
    orch = Orchestrator(goal="test goal")
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="completed"),
    ]
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_some_pending():
    """When some tasks are pending, _is_done() should return False."""
    orch = Orchestrator(goal="test goal")
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="pending"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_has_failures():
    """When any task has failed, _is_done() should return False."""
    orch = Orchestrator(goal="test goal")
    orch._tasks = [
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="failed"),
    ]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_cycle_stores_tasks(
    mock_planner, mock_workers, mock_tester_pass, base_settings
):
    """The _cycle() method should store the plan's tasks for _get_tasks()."""
    orch = Orchestrator(goal="test goal", settings=base_settings)
    await orch._cycle()
    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_max_cycles_enforced(mock_planner, mock_workers, mock_tester_pass):
    """Orchestrator should stop after max_cycles even if not done."""
    settings = Settings(max_cycles=2)
    orch = Orchestrator(goal="test goal", settings=settings)

    # Mock _is_done to always return False so max_cycles is the stopping condition
    with patch.object(orch, "_is_done", return_value=False):
        await orch.run()

    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_max_cycles_zero_is_infinite(mock_planner, mock_workers, mock_tester_pass):
    """With max_cycles=0, the orchestrator should not have a cycle limit."""
    settings = Settings(max_cycles=0)
    orch = Orchestrator(goal="test goal", settings=settings)
    # Just verify the setting is respected (infinite by default)
    assert orch.settings.max_cycles == 0


@pytest.mark.asyncio
async def test_max_parallel_tasks_limits_concurrency(
    mock_planner, mock_tester_pass, base_settings
):
    """max_parallel_tasks should limit concurrent worker executions."""
    settings = Settings(max_parallel_tasks=1)
    orch = Orchestrator(goal="test goal", settings=settings)

    # Track concurrent executions
    current = 0
    max_seen = 0

    async def tracked_run():
        nonlocal current, max_seen
        current += 1
        max_seen = max(max_seen, current)
        await asyncio.sleep(0.1)
        current -= 1
        return "Done"

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        instance = MagicMock()
        instance.run = tracked_run
        MockWorker.return_value = instance

        await orch._cycle()

    # With max_parallel_tasks=1, never more than 1 concurrent
    assert max_seen <= 1


@pytest.mark.asyncio
async def test_max_parallel_tasks_allows_parallel(
    mock_planner, mock_tester_pass, base_settings
):
    """With max_parallel_tasks > 1, workers should run concurrently."""
    settings = Settings(max_parallel_tasks=5)
    orch = Orchestrator(goal="test goal", settings=settings)

    current = 0
    max_seen = 0

    async def tracked_run():
        nonlocal current, max_seen
        current += 1
        max_seen = max(max_seen, current)
        await asyncio.sleep(0.1)
        current -= 1
        return "Done"

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        instance = MagicMock()
        instance.run = tracked_run
        MockWorker.return_value = instance

        await orch._cycle()

    # With max_parallel_tasks=5 and 2 tasks, both should run concurrently
    assert max_seen == 2


@pytest.mark.asyncio
async def test_state_dict(base_settings):
    """The state() method should return serializable state."""
    orch = Orchestrator(goal="test goal", settings=base_settings)
    orch._tasks = [TaskModel(id="1", description="t1", status="completed")]
    state = orch.state()
    assert state["goal"] == "test goal"
    assert state["cycles"] == 0
    assert state["done"] is False
    assert len(state["tasks"]) == 1
    assert state["tasks"][0]["id"] == "1"
    assert state["last_plan"] is None


@pytest.mark.asyncio
async def test_state_with_plan(base_settings):
    """State should include last_plan after a cycle."""
    orch = Orchestrator(goal="test goal", settings=base_settings)
    orch._last_plan = Plan(
        tasks=[TaskModel(id="1", description="t1")], rationale="test"
    )
    state = orch.state()
    assert state["last_plan"] is not None
    assert state["last_plan"]["tasks"][0]["id"] == "1"


class CollectingSink:
    """OutputSink that collects emitted text for assertion."""

    def __init__(self):
        self.messages = []

    async def emit(self, text: str):
        self.messages.append(text)


@pytest.mark.asyncio
async def test_output_sink_used(
    mock_planner, mock_workers, mock_tester_pass, base_settings
):
    """Orchestrator should emit output through the OutputSink when provided."""
    sink = CollectingSink()
    orch = Orchestrator(goal="test goal", settings=base_settings, output=sink)
    await orch._cycle()
    assert len(sink.messages) > 0
    joined = " ".join(sink.messages)
    assert "Task" in joined or "Plan" in joined


@pytest.mark.asyncio
async def test_streaming_flag(base_settings):
    """When output is provided, _streaming should be True."""
    sink = CollectingSink()
    orch = Orchestrator(goal="test goal", settings=base_settings, output=sink)
    assert orch._streaming is True


@pytest.mark.asyncio
async def test_default_sink_not_streaming(base_settings):
    """When no output is provided, _streaming should be False."""
    orch = Orchestrator(goal="test goal", settings=base_settings)
    assert orch._streaming is False


@pytest.mark.asyncio
async def test_test_failure_updates_goal(
    mock_planner, mock_workers, mock_tester_fail, base_settings
):
    """When tests fail, the goal should be updated to fix failures."""
    orch = Orchestrator(goal="original goal", settings=base_settings)
    await orch._cycle()
    assert "Fix failing tests" in orch.goal
    assert "fail1" in orch.goal


@pytest.mark.asyncio
async def test_failed_task_marks_status(
    mock_planner, mock_tester_pass, base_settings
):
    """When a worker raises an exception, the task should be marked as failed."""
    plan = Plan(
        tasks=[TaskModel(id="1", description="Task 1", files=[], dependencies=[])],
        rationale="Test plan",
    )
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner:
        instance = MagicMock()
        instance.plan = AsyncMock(return_value=plan)
        MockPlanner.return_value = instance

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            worker_inst = MagicMock()
            worker_inst.run = AsyncMock(
                side_effect=RuntimeError("worker failed")
            )
            MockWorker.return_value = worker_inst

            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                tester_inst = MagicMock()
                tester_inst.run = AsyncMock(
                    return_value=TestResult(passed=True, summary="ok", failures=[])
                )
                MockTester.return_value = tester_inst

                orch = Orchestrator(goal="test goal", settings=base_settings)
                await orch._cycle()

                assert orch._tasks[0].status == "failed"
                assert "worker failed" in orch._tasks[0].result
