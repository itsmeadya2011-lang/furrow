from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestOrchestrator:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(max_cycles=0, workspace="/tmp")
        return client

    @pytest.fixture
    def orchestrator(self, mock_client):
        return Orchestrator(goal="Test goal", client=mock_client, max_cycles=5)

    def test_get_tasks_returns_persisted_tasks(self, orchestrator):
        task1 = TaskModel(id="1", description="Task 1")
        task2 = TaskModel(id="2", description="Task 2")
        orchestrator.tasks = [task1, task2]
        assert orchestrator._get_tasks() == [task1, task2]

    def test_is_done_returns_true_when_all_tasks_completed(self, orchestrator):
        task1 = TaskModel(id="1", description="Task 1", status="completed")
        task2 = TaskModel(id="2", description="Task 2", status="completed")
        orchestrator.tasks = [task1, task2]
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_task_failed(self, orchestrator):
        task1 = TaskModel(id="1", description="Task 1", status="completed")
        task2 = TaskModel(id="2", description="Task 2", status="failed")
        orchestrator.tasks = [task1, task2]
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_no_tasks(self, orchestrator):
        orchestrator.tasks = []
        assert orchestrator._is_done() is False

    def test_is_done_returns_false_when_tasks_pending(self, orchestrator):
        task1 = TaskModel(id="1", description="Task 1", status="completed")
        task2 = TaskModel(id="2", description="Task 2", status="pending")
        orchestrator.tasks = [task1, task2]
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_max_cycles_enforcement(self, mock_client):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, max_cycles=2)

        mock_plan = Plan(
            tasks=[TaskModel(id="1", description="Task 1")],
            rationale="Test rationale"
        )

        with patch.object(orchestrator.planner, 'plan', new_callable=AsyncMock) as mock_plan_method:
            mock_plan_method.return_value = mock_plan

            with patch('furrow.core.orchestrator.WorkerAgent') as mock_worker:
                mock_worker_instance = AsyncMock()
                # Make worker fail so task status becomes "failed", preventing _is_done() from returning True
                mock_worker_instance.run.side_effect = Exception("Worker failed")
                mock_worker.return_value = mock_worker_instance

                with patch('furrow.core.orchestrator.TesterAgent') as mock_tester:
                    mock_tester_instance = AsyncMock()
                    mock_tester_instance.run.return_value = type('TestResult', (), {
                        'passed': False,
                        'summary': 'Tests failed',
                        'failures': ['Test failed']
                    })()
                    mock_tester.return_value = mock_tester_instance

                    await orchestrator.run()

        assert orchestrator.cycles == 2

    @pytest.mark.asyncio
    async def test_tasks_merged_correctly_across_cycles(self, mock_client):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, max_cycles=3)

        plan1 = Plan(
            tasks=[TaskModel(id="1", description="Task 1"), TaskModel(id="2", description="Task 2")],
            rationale="Cycle 1"
        )
        plan2 = Plan(
            tasks=[TaskModel(id="2", description="Task 2 updated"), TaskModel(id="3", description="Task 3")],
            rationale="Cycle 2"
        )
        plan3 = Plan(
            tasks=[TaskModel(id="1", description="Task 1"), TaskModel(id="3", description="Task 3"), TaskModel(id="4", description="Task 4")],
            rationale="Cycle 3"
        )

        plans = [plan1, plan2, plan3]

        with patch.object(orchestrator.planner, 'plan', new_callable=AsyncMock) as mock_plan_method:
            mock_plan_method.side_effect = plans

            with patch('furrow.core.orchestrator.WorkerAgent') as mock_worker:
                # Make worker fail so tasks get "failed" status, preventing _is_done() from returning True
                mock_worker_instance = AsyncMock()
                mock_worker_instance.run.side_effect = Exception("Worker failed")
                mock_worker.return_value = mock_worker_instance

                with patch('furrow.core.orchestrator.TesterAgent') as mock_tester:
                    mock_tester_instance = AsyncMock()
                    mock_tester_instance.run.return_value = type('TestResult', (), {
                        'passed': False,
                        'summary': 'Tests failed',
                        'failures': ['Test failed']
                    })()
                    mock_tester.return_value = mock_tester_instance

                    await orchestrator.run()

        task_ids = [t.id for t in orchestrator.tasks]
        assert task_ids == ["1", "2", "3", "4"]
        assert len(orchestrator.tasks) == 4

    def test_stop_sets_stopped_flag(self, orchestrator):
        assert orchestrator._stopped is False
        orchestrator.stop()
        assert orchestrator._stopped is True

    @pytest.mark.asyncio
    async def test_stop_stops_orchestrator_early(self, mock_client):
        orchestrator = Orchestrator(goal="Test goal", client=mock_client, max_cycles=10)

        mock_plan = Plan(
            tasks=[TaskModel(id="1", description="Task 1")],
            rationale="Test rationale"
        )

        with patch.object(orchestrator.planner, 'plan', new_callable=AsyncMock) as mock_plan_method:
            mock_plan_method.return_value = mock_plan

            with patch('furrow.core.orchestrator.WorkerAgent') as mock_worker:
                mock_worker_instance = AsyncMock()
                mock_worker_instance.run.return_value = "Done"
                mock_worker.return_value = mock_worker_instance

                with patch('furrow.core.orchestrator.TesterAgent') as mock_tester:
                    mock_tester_instance = AsyncMock()
                    mock_tester_instance.run.return_value = type('TestResult', (), {
                        'passed': False,
                        'summary': 'Tests failed',
                        'failures': ['Test failed']
                    })()
                    mock_tester.return_value = mock_tester_instance

                    orchestrator.stop()
                    await orchestrator.run()

        assert orchestrator.cycles == 0