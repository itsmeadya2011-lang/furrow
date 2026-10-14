from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.settings = Settings(
        provider=Provider.ANTHROPIC,
        max_cycles=0,
        max_parallel_tasks=5,
        workspace=".",
    )
    return client


@pytest.fixture
def orchestrator(mock_client, tmp_path):
    # Use a temporary state path to avoid loading previous state
    state_path = tmp_path / "state.json"
    return Orchestrator(goal="Test goal", client=mock_client, state_path=state_path)


class TestOrchestratorIsDone:
    def test_is_done_returns_true_when_no_tasks(self, orchestrator):
        orchestrator._tasks = []
        assert orchestrator._is_done() is True

    def test_is_done_returns_true_when_all_tasks_completed_and_tests_passed(self, orchestrator):
        orchestrator._tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]
        orchestrator._last_test_result = TestResult(passed=True, summary="All passed", failures=[])
        assert orchestrator._is_done() is True

    def test_is_done_returns_true_when_all_tasks_completed_and_last_test_result_none(
        self, orchestrator
    ):
        orchestrator._tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
        ]
        orchestrator._last_test_result = None
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_task_failed(self, orchestrator):
        orchestrator._tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="failed"),
        ]
        orchestrator._last_test_result = TestResult(passed=True, summary="All passed", failures=[])
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_not_all_tasks_completed(self, orchestrator):
        orchestrator._tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="pending"),
        ]
        orchestrator._last_test_result = TestResult(passed=True, summary="All passed", failures=[])
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_all_tasks_completed_but_tests_failed(self, orchestrator):
        orchestrator._tasks = [
            TaskModel(id="1", description="Task 1", status="completed"),
            TaskModel(id="2", description="Task 2", status="completed"),
        ]
        orchestrator._last_test_result = TestResult(
            passed=False, summary="Tests failed", failures=["test_foo failed"]
        )
        assert orchestrator._is_done() is False


class TestOrchestratorMaxCycles:
    @pytest.mark.asyncio
    async def test_max_cycles_enforcement(self, mock_client, tmp_path):
        mock_client.settings.max_cycles = 2
        state_path = tmp_path / "state.json"

        with patch("furrow.core.orchestrator.PlannerAgent") as mock_planner_class:
            mock_planner = AsyncMock()
            mock_planner.plan.return_value = MagicMock(
                tasks=[TaskModel(id="1", description="Task 1")], rationale="test"
            )
            mock_planner_class.return_value = mock_planner

            with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
                mock_worker = AsyncMock()
                mock_worker.run.return_value = "done"
                mock_worker_class.return_value = mock_worker

                with patch("furrow.core.orchestrator.TesterAgent") as mock_tester_class:
                    mock_tester = AsyncMock()
                    mock_tester.run.return_value = TestResult(
                        passed=False, summary="failed", failures=["fail"]
                    )
                    mock_tester_class.return_value = mock_tester

                    orchestrator = Orchestrator(
                        goal="Test goal", client=mock_client, state_path=state_path
                    )
                    await orchestrator.run()

                    assert orchestrator.cycles == 2
                    assert mock_planner.plan.call_count == 2
