"""Tests for the Orchestrator core loop logic."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def make_mock_client(max_cycles: int = 0, max_parallel: int = 5) -> MagicMock:
    """Create a mock LLMClient with settings for testing."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.max_cycles = max_cycles
    client.settings.max_parallel_tasks = max_parallel
    client.settings.workspace = MagicMock()
    client.settings.workspace.exists = MagicMock(return_value=False)
    return client


def make_task(id: str = "1", status: str = "pending") -> TaskModel:
    return TaskModel(id=id, description=f"task {id}", status=status)


class TestIsDone:
    def test_no_tasks_is_done(self):
        """When no tasks are planned, the goal is considered complete."""
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = []
        orch.last_test_result = None
        assert orch._is_done() is True

    def test_all_tasks_completed_passed_tests(self):
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = [make_task("1", "completed"), make_task("2", "completed")]
        orch.last_test_result = TestResult(passed=True, summary="ok")
        assert orch._is_done() is True

    def test_task_failed_not_done(self):
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = [make_task("1", "completed"), make_task("2", "failed")]
        orch.last_test_result = TestResult(passed=False, summary="broken")
        # Some tasks completed, some failed → not done
        assert orch._is_done() is False

    def test_all_tasks_completed_but_tests_failed(self):
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = [make_task("1", "completed"), make_task("2", "completed")]
        orch.last_test_result = TestResult(passed=False, summary="failures!")
        # All tasks completed but tests failed → not done (should retry)
        assert orch._is_done() is False

    def test_incomplete_tasks_not_done(self):
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = [make_task("1", "completed"), make_task("2", "pending")]
        orch.last_test_result = TestResult(passed=True, summary="ok")
        assert orch._is_done() is False

    def test_all_tasks_failed_stop(self):
        """If every task failed and none completed, stop to avoid infinite loop."""
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = [make_task("1", "failed"), make_task("2", "failed")]
        orch.last_test_result = TestResult(passed=False, summary="all failed")
        assert orch._is_done() is True

    def test_no_test_result_all_completed(self):
        """If no test was run yet but all tasks completed, consider done."""
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = [make_task("1", "completed")]
        orch.last_test_result = None
        assert orch._is_done() is True


class TestGetTasks:
    def test_get_tasks_returns_stored_tasks(self):
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        tasks = [make_task("1"), make_task("2")]
        orch.tasks = tasks
        assert orch._get_tasks() == tasks

    def test_get_tasks_default_empty(self):
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        assert orch._get_tasks() == []


class TestStatus:
    def test_status_idle(self):
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = []
        assert orch.status == "idle"

    def test_status_with_tasks(self):
        orch = Orchestrator(goal="test goal", client=make_mock_client())
        orch.tasks = [make_task("1", "completed"), make_task("2", "pending")]
        orch.cycles = 1
        assert "1/2 completed" in orch.status
        assert "cycle 1" in orch.status


class TestMaxCycles:
    @pytest.mark.asyncio
    async def test_max_cycles_stops_loop(self):
        """Orchestrator should stop when max_cycles is reached."""
        client = make_mock_client(max_cycles=3)

        orch = Orchestrator(goal="test goal", client=client)
        call_count = 0

        async def mock_cycle():
            nonlocal call_count
            call_count += 1

        orch._cycle = mock_cycle
        orch._is_done = MagicMock(return_value=False)

        await orch.run()

        assert call_count == 3  # Only 3 cycles allowed


class TestExecuteTasks:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """max_parallel_tasks should limit concurrent worker execution."""
        orch = Orchestrator(goal="test goal", client=make_mock_client(max_parallel=2))
        tasks = [make_task(str(i)) for i in range(5)]

        current = 0
        max_concurrent = 0

        async def mock_worker_run(task=None):
            nonlocal current, max_concurrent
            current += 1
            max_concurrent = max(max_concurrent, current)
            await asyncio.sleep(0.05)
            current -= 1
            return "done"

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = mock_worker_run
            results = await orch._execute_tasks(tasks)

        assert max_concurrent <= 2
        assert len(results) == 5


class TestOrchestratorRun:
    @pytest.mark.asyncio
    async def test_run_stops_on_done(self):
        """The run loop should stop when _is_done returns True."""
        client = make_mock_client()
        orch = Orchestrator(goal="test goal", client=client)

        orch._cycle = AsyncMock()
        orch._is_done = MagicMock(return_value=True)

        await orch.run()

        orch._cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_multiple_cycles_until_done(self):
        """The run loop should continue until _is_done returns True."""
        client = make_mock_client(max_cycles=100)
        orch = Orchestrator(goal="test goal", client=client)

        call_count = 0
        done_values = [False, False, True]  # Done after 3 cycles

        async def mock_cycle():
            nonlocal call_count
            call_count += 1

        def mock_is_done():
            return done_values[call_count - 1] if call_count - 1 < len(done_values) else True

        orch._cycle = mock_cycle
        orch._is_done = mock_is_done

        await orch.run()

        assert call_count == 3  # Should run 3 cycles before stopping
