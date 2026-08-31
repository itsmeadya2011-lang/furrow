"""Tests for the Orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


@pytest.fixture
def mock_client():
    return MagicMock(spec=LLMClient)


@pytest.fixture
def sample_plan():
    return Plan(
        tasks=[
            TaskModel(id="1", description="Implement feature A", files=["src/a.py"]),
            TaskModel(id="2", description="Implement feature B", files=["src/b.py"]),
        ],
        rationale="Two independent features",
    )


@pytest.fixture
def orchestrator(mock_client):
    return Orchestrator(goal="Test goal", client=mock_client)


class TestOrchestrator:
    def test_init(self, orchestrator):
        assert orchestrator.goal == "Test goal"
        assert orchestrator.cycles == 0
        assert orchestrator.plan is None

    def test_init_with_max_cycles(self, mock_client):
        orch = Orchestrator(goal="Test", client=mock_client, max_cycles=5)
        assert orch.max_cycles == 5

    def test_get_tasks_empty_when_no_plan(self, orchestrator):
        assert orchestrator._get_tasks() == []

    def test_get_tasks_returns_plan_tasks(self, orchestrator, sample_plan):
        orchestrator.plan = sample_plan
        tasks = orchestrator._get_tasks()
        assert len(tasks) == 2
        assert tasks[0].id == "1"
        assert tasks[1].id == "2"

    def test_is_done_no_tasks(self, orchestrator):
        assert orchestrator._is_done() is True

    def test_is_done_all_completed(self, orchestrator, sample_plan):
        orchestrator.plan = sample_plan
        for task in sample_plan.tasks:
            task.status = "completed"
        assert orchestrator._is_done() is True

    def test_is_done_with_failures(self, orchestrator, sample_plan):
        orchestrator.plan = sample_plan
        sample_plan.tasks[0].status = "completed"
        sample_plan.tasks[1].status = "failed"
        assert orchestrator._is_done() is False

    def test_is_done_partial_completion(self, orchestrator, sample_plan):
        orchestrator.plan = sample_plan
        sample_plan.tasks[0].status = "completed"
        sample_plan.tasks[1].status = "pending"
        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_cycle_stores_plan(self, orchestrator, sample_plan):
        orchestrator.planner.plan = AsyncMock(return_value=sample_plan)

        mock_worker = AsyncMock(return_value="Task completed")
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = mock_worker
            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                mock_tester = AsyncMock()
                mock_tester.run = AsyncMock(
                    return_value=TestResult(passed=True, summary="All good")
                )
                MockTester.return_value = mock_tester

                await orchestrator._cycle()

        assert orchestrator.plan is not None
        assert len(orchestrator.plan.tasks) == 2

    @pytest.mark.asyncio
    async def test_run_stops_at_max_cycles(self, mock_client):
        orch = Orchestrator(goal="Test", client=mock_client, max_cycles=2)

        sample_plan = Plan(
            tasks=[TaskModel(id="1", description="Task 1")],
            rationale="One task",
        )
        orch.planner.plan = AsyncMock(return_value=sample_plan)

        mock_worker = AsyncMock(return_value="Done")
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = mock_worker
            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                mock_tester = AsyncMock()
                mock_tester.run = AsyncMock(
                    return_value=TestResult(passed=False, summary="Failed", failures=["err"])
                )
                MockTester.return_value = mock_tester

                await orch.run()

        assert orch.cycles == 2