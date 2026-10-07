import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


@pytest.fixture
def settings():
    return Settings(
        provider=Provider.ANTHROPIC,
        anthropic_api_key="test-key",
        max_cycles=5,
        max_parallel_tasks=2,
    )


@pytest.fixture
def mock_client(settings):
    client = MagicMock(spec=LLMClient)
    client.settings = settings
    return client


@pytest.fixture
def orchestrator(mock_client):
    return Orchestrator(goal="test goal", client=mock_client)


class TestOrchestrator:
    def test_get_tasks_returns_internal_tasks_list(self, orchestrator):
        task1 = TaskModel(id="1", description="task 1")
        task2 = TaskModel(id="2", description="task 2")
        orchestrator.tasks = [task1, task2]
        tasks = orchestrator._get_tasks()
        assert tasks is orchestrator.tasks
        assert len(tasks) == 2
        assert tasks[0].id == "1"
        assert tasks[1].id == "2"

    def test_is_done_returns_true_when_no_tasks(self, orchestrator):
        orchestrator.tasks = []
        assert orchestrator._is_done() is True

    def test_is_done_returns_true_when_all_completed_and_tests_passed(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        orchestrator.last_test_passed = True
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_task_failed(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ]
        orchestrator.last_test_passed = True
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_tasks_not_completed(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="pending"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        orchestrator.last_test_passed = True
        assert orchestrator._is_done() is False

    def test_is_done_respects_max_cycles(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="pending"),
        ]
        orchestrator.cycles = 5
        orchestrator.settings.max_cycles = 5
        assert orchestrator._is_done() is True

    def test_is_done_false_when_under_max_cycles(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="pending"),
        ]
        orchestrator.cycles = 3
        orchestrator.settings.max_cycles = 5
        assert orchestrator._is_done() is False

    def test_is_done_max_cycles_zero_no_limit(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="pending"),
        ]
        orchestrator.cycles = 100
        orchestrator.settings.max_cycles = 0
        assert orchestrator._is_done() is False

    def test_merge_tasks_updates_existing_by_id(self, orchestrator):
        existing_task = TaskModel(id="1", description="old", files=["old.py"])
        orchestrator.tasks = [existing_task]
        new_task = TaskModel(id="1", description="new", files=["new.py"], dependencies=["2"])
        orchestrator._merge_tasks([new_task])
        assert len(orchestrator.tasks) == 1
        assert orchestrator.tasks[0].description == "new"
        assert orchestrator.tasks[0].files == ["new.py"]
        assert orchestrator.tasks[0].dependencies == ["2"]
        assert orchestrator.tasks[0].id == "1"

    def test_merge_tasks_appends_new_tasks(self, orchestrator):
        orchestrator.tasks = [TaskModel(id="1", description="task 1")]
        new_tasks = [
            TaskModel(id="2", description="task 2"),
            TaskModel(id="3", description="task 3"),
        ]
        orchestrator._merge_tasks(new_tasks)
        assert len(orchestrator.tasks) == 3
        assert orchestrator.tasks[1].id == "2"
        assert orchestrator.tasks[2].id == "3"

    def test_merge_tasks_mixed_existing_and_new(self, orchestrator):
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ]
        new_tasks = [
            TaskModel(id="2", description="updated task 2"),
            TaskModel(id="3", description="task 3"),
        ]
        orchestrator._merge_tasks(new_tasks)
        assert len(orchestrator.tasks) == 3
        assert orchestrator.tasks[0].id == "1"
        assert orchestrator.tasks[1].id == "2"
        assert orchestrator.tasks[1].description == "updated task 2"
        assert orchestrator.tasks[2].id == "3"

    @pytest.mark.asyncio
    async def test_task_status_updates_on_worker_success(self, orchestrator, mock_client):
        task = TaskModel(id="1", description="test task")
        orchestrator.tasks = [task]
        orchestrator.last_test_passed = True

        # Mock the planner's LLM call to return a valid plan
        plan_data = {
            "tasks": [{"id": "1", "description": "test task", "files": [], "dependencies": []}],
            "rationale": "test rationale"
        }
        mock_client.complete = AsyncMock(return_value=json.dumps(plan_data))

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.run.return_value = "success"
            mock_worker_class.return_value = mock_worker

            with patch("furrow.core.orchestrator.TesterAgent") as mock_tester_class:
                mock_tester = AsyncMock()
                mock_tester.run.return_value = TestResult(passed=True, summary="all good", failures=[])
                mock_tester_class.return_value = mock_tester

                await orchestrator._cycle()

                assert task.status == "completed"
                assert task.result == "success"

    @pytest.mark.asyncio
    async def test_task_status_updates_on_worker_failure(self, orchestrator, mock_client):
        task = TaskModel(id="1", description="test task")
        orchestrator.tasks = [task]
        orchestrator.last_test_passed = True

        # Mock the planner's LLM call to return a valid plan
        plan_data = {
            "tasks": [{"id": "1", "description": "test task", "files": [], "dependencies": []}],
            "rationale": "test rationale"
        }
        mock_client.complete = AsyncMock(return_value=json.dumps(plan_data))

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.run.side_effect = Exception("worker failed")
            mock_worker_class.return_value = mock_worker

            with patch("furrow.core.orchestrator.TesterAgent") as mock_tester_class:
                mock_tester = AsyncMock()
                mock_tester.run.return_value = TestResult(passed=True, summary="all good", failures=[])
                mock_tester_class.return_value = mock_tester

                await orchestrator._cycle()

                assert task.status == "failed"
                assert "worker failed" in task.result

    @pytest.mark.asyncio
    async def test_cycle_calls_planner_worker_tester(self, orchestrator, mock_client):
        orchestrator.tasks = []

        # Mock the planner's LLM call to return a valid plan
        plan_data = {
            "tasks": [{"id": "1", "description": "test task", "files": [], "dependencies": []}],
            "rationale": "test rationale"
        }
        mock_client.complete = AsyncMock(return_value=json.dumps(plan_data))

        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_class:
            mock_worker = AsyncMock()
            mock_worker.run.return_value = "done"
            mock_worker_class.return_value = mock_worker

            with patch("furrow.core.orchestrator.TesterAgent") as mock_tester_class:
                mock_tester = AsyncMock()
                mock_tester.run.return_value = TestResult(passed=True, summary="passed", failures=[])
                mock_tester_class.return_value = mock_tester

                await orchestrator._cycle()

                mock_worker.run.assert_called_once()
                # The task passed to tester should be the updated task (completed with result)
                called_args = mock_tester.run.call_args
                assert called_args[0][0] == "test goal"
                test_tasks = called_args[0][1]
                assert len(test_tasks) == 1
                assert test_tasks[0].id == "1"
                assert test_tasks[0].status == "completed"
                assert test_tasks[0].result == "done"
                assert orchestrator.last_test_passed is True
                assert orchestrator.test_result.passed is True


class TestOrchestratorInitialization:
    def test_initializes_with_default_client(self):
        with patch("furrow.core.orchestrator.LLMClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client.settings = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test")
            mock_client_class.return_value = mock_client
            orchestrator = Orchestrator(goal="test")
            assert orchestrator.goal == "test"
            assert orchestrator.client is mock_client
            assert orchestrator.cycles == 0
            assert orchestrator.tasks == []
            assert orchestrator.test_result is None
            assert orchestrator.last_test_passed is False

    def test_initializes_with_provided_client(self, mock_client):
        orchestrator = Orchestrator(goal="test", client=mock_client)
        assert orchestrator.client is mock_client