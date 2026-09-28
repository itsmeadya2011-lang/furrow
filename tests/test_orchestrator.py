"""Tests for the Orchestrator: task tracking, termination logic, output callback."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def mock_client():
    """Create a mock LLMClient that won't make real API calls."""
    client = MagicMock()
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')
    client.settings = MagicMock()
    client.settings.planner_model = "haiku"
    client.settings.worker_model = "sonnet"
    client.settings.tester_model = "sonnet"
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = 0
    client.read_file = AsyncMock(return_value="# original content")
    client.write_file = AsyncMock()
    client.list_files = MagicMock(return_value=[])
    return client


@pytest.fixture
def completed_plan():
    return Plan(
        tasks=[
            TaskModel(id="1", description="task one", status="pending"),
            TaskModel(id="2", description="task two", status="pending"),
        ],
        rationale="test plan",
    )


class TestOrchestratorInit:
    def test_basic_init(self, mock_client):
        orch = Orchestrator(goal="test goal", client=mock_client)
        assert orch.goal == "test goal"
        assert orch.client is mock_client
        assert orch.cycles == 0
        assert orch.tasks == []
        assert orch.last_test_result is None

    def test_init_with_output_callback(self, mock_client):
        received = []

        def callback(text: str) -> None:
            received.append(text)

        orch = Orchestrator(
            goal="test",
            client=mock_client,
            output_callback=callback,
        )
        assert orch.output_callback is callback

    def test_init_without_output_callback(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        assert orch.output_callback is None


class TestIsDone:
    def test_no_tasks_is_done(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = []
        assert orch._is_done() is True

    def test_all_completed_tests_pass_is_done(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="completed"),
        ]
        orch.last_test_result = TestResult(passed=True, summary="all passed")
        assert orch._is_done() is True

    def test_failed_tasks_with_failing_tests_not_done(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [TaskModel(id="1", description="task1", status="failed")]
        orch.last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
        assert orch._is_done() is False

    def test_completed_tasks_with_failing_tests_not_done(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [TaskModel(id="1", description="task1", status="completed")]
        orch.last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
        assert orch._is_done() is False

    def test_partial_completion_not_done(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [
            TaskModel(id="1", description="task1", status="completed"),
            TaskModel(id="2", description="task2", status="pending"),
        ]
        orch.last_test_result = TestResult(passed=True, summary="pass")
        assert orch._is_done() is False

    def test_no_test_result_treated_as_passing(self, mock_client):
        """When tests haven't run yet, passing is assumed (first cycle)."""
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [TaskModel(id="1", description="task1", status="completed")]
        orch.last_test_result = None
        assert orch._is_done() is True


class TestMaxCycles:
    def test_max_cycles_stops_loop(self, mock_client):
        """When max_cycles is reached, _is_done returns True."""
        orch = Orchestrator(goal="test", client=mock_client)
        orch.cycles = 3
        orch.tasks = [TaskModel(id="1", description="task1", status="completed")]
        orch.last_test_result = TestResult(passed=True, summary="ok")
        with patch("furrow.core.orchestrator.settings") as mock_settings:
            mock_settings.max_cycles = 3
            mock_settings.max_parallel_tasks = 5
            assert orch._is_done() is True

    def test_max_cycles_zero_means_unlimited(self, mock_client):
        """max_cycles=0 means unlimited (default)."""
        orch = Orchestrator(goal="test", client=mock_client)
        orch.cycles = 100
        orch.tasks = [TaskModel(id="1", description="task1", status="completed")]
        orch.last_test_result = TestResult(passed=True, summary="ok")
        with patch("furrow.core.orchestrator.settings") as mock_settings:
            mock_settings.max_cycles = 0
            mock_settings.max_parallel_tasks = 5
            assert orch._is_done() is True  # all completed + tests pass


class TestEmit:
    def test_emit_with_callback(self, mock_client):
        received = []
        orch = Orchestrator(goal="test", client=mock_client, output_callback=received.append)
        orch._emit("hello world")
        assert len(received) == 1
        assert "hello world" in received[0]

    def test_emit_without_callback(self, mock_client, capsys):
        orch = Orchestrator(goal="test", client=mock_client)
        orch._emit("visible output")
        captured = capsys.readouterr()
        assert "visible output" in captured.out


class TestCycle:
    @pytest.mark.asyncio
    async def test_cycle_stores_tasks(self, mock_client, completed_plan):
        """After _cycle, self.tasks should be populated from the plan."""
        orch = Orchestrator(goal="test", client=mock_client)
        mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')

        with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, patch(
            "furrow.core.orchestrator.WorkerAgent"
        ) as MockWorker, patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            mock_planner = MagicMock()
            mock_planner.plan = AsyncMock(return_value=completed_plan)
            MockPlanner.return_value = mock_planner

            mock_worker = MagicMock()
            mock_worker.run = AsyncMock(return_value="worker result")
            MockWorker.return_value = mock_worker

            mock_tester = MagicMock()
            mock_tester.run = AsyncMock(
                return_value=TestResult(passed=True, summary="all passed")
            )
            MockTester.return_value = mock_tester

            await orch._cycle()

        assert len(orch.tasks) == 2
        assert orch.tasks[0].id == "1"
        assert orch.tasks[0].status == "completed"
        assert orch.last_test_result.passed is True


class TestGetTasks:
    def test_get_tasks_returns_stored_tasks(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        tasks = [TaskModel(id="1", description="task1")]
        orch.tasks = tasks
        assert orch._get_tasks() is tasks
