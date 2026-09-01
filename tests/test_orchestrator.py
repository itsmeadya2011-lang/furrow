from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestOrchestrator:
    def test_init_default(self):
        orch = Orchestrator(goal="test goal")
        assert orch.goal == "test goal"
        assert orch.cycles == 0
        assert orch.client is not None
        assert orch.planner is not None

    def test_init_with_custom_client(self, mock_llm_client):
        orch = Orchestrator(goal="test goal", client=mock_llm_client)
        assert orch.client == mock_llm_client

    def test_is_done_all_completed(self, mock_llm_client):
        orch = Orchestrator(goal="test", client=mock_llm_client)
        orch.tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        assert orch._is_done() is True

    def test_is_done_with_pending(self, mock_llm_client):
        orch = Orchestrator(goal="test", client=mock_llm_client)
        orch.tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ]
        assert orch._is_done() is False

    def test_is_done_with_failed(self, mock_llm_client):
        orch = Orchestrator(goal="test", client=mock_llm_client)
        orch.tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed"),
        ]
        assert orch._is_done() is False

    def test_is_done_empty_tasks(self, mock_llm_client):
        orch = Orchestrator(goal="test", client=mock_llm_client)
        orch.tasks = []
        assert orch._is_done() is True

    def test_is_done_all_pending(self, mock_llm_client):
        orch = Orchestrator(goal="test", client=mock_llm_client)
        orch.tasks = [
            TaskModel(id="1", description="t1", status="pending"),
            TaskModel(id="2", description="t2", status="pending"),
        ]
        assert orch._is_done() is False

    @pytest.mark.asyncio
    async def test_cycle_stores_tasks(self, mock_llm_client, sample_plan):
        mock_llm_client.plan = AsyncMock(return_value=sample_plan)
        orch = Orchestrator(goal="test", client=mock_llm_client)

        with patch.object(orch.planner, "plan", return_value=sample_plan):
            with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
                mock_worker = MagicMock()
                mock_worker.run = AsyncMock(return_value="done")
                MockWorker.return_value = mock_worker

                with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                    mock_tester = MagicMock()
                    mock_tester.run = AsyncMock(
                        return_value=MagicMock(passed=True, summary="ok", failures=[])
                    )
                    MockTester.return_value = mock_tester

                    await orch._cycle()

        assert orch.tasks is not None
        assert len(orch.tasks) == 2
        assert orch.tasks[0].id == "1"
        assert orch.tasks[1].id == "2"

    @pytest.mark.asyncio
    async def test_cycle_handles_empty_plan(self, mock_llm_client):
        empty_plan = Plan(tasks=[], rationale="nothing to do")
        orch = Orchestrator(goal="test", client=mock_llm_client)

        with patch.object(orch.planner, "plan", return_value=empty_plan):
            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                mock_tester = MagicMock()
                mock_tester.run = AsyncMock(
                    return_value=MagicMock(passed=True, summary="ok", failures=[])
                )
                MockTester.return_value = mock_tester

                await orch._cycle()

        assert orch.tasks == []

    @pytest.mark.asyncio
    async def test_cycle_updates_task_status_on_success(self, mock_llm_client, sample_plan):
        orch = Orchestrator(goal="test", client=mock_llm_client)

        with patch.object(orch.planner, "plan", return_value=sample_plan):
            with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
                mock_worker = MagicMock()
                mock_worker.run = AsyncMock(return_value="task completed successfully")
                MockWorker.return_value = mock_worker

                with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                    mock_tester = MagicMock()
                    mock_tester.run = AsyncMock(
                        return_value=MagicMock(passed=True, summary="ok", failures=[])
                    )
                    MockTester.return_value = mock_tester

                    await orch._cycle()

        for task in orch.tasks:
            assert task.status == "completed"
            assert task.result == "task completed successfully"

    @pytest.mark.asyncio
    async def test_cycle_updates_task_status_on_failure(self, mock_llm_client, sample_plan):
        orch = Orchestrator(goal="test", client=mock_llm_client)

        with patch.object(orch.planner, "plan", return_value=sample_plan):
            with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
                mock_worker = MagicMock()
                mock_worker.run = AsyncMock(side_effect=Exception("task failed"))
                MockWorker.return_value = mock_worker

                with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                    mock_tester = MagicMock()
                    mock_tester.run = AsyncMock(
                        return_value=MagicMock(passed=True, summary="ok", failures=[])
                    )
                    MockTester.return_value = mock_tester

                    await orch._cycle()

        for task in orch.tasks:
            assert task.status == "failed"
            assert "task failed" in task.result

    @pytest.mark.asyncio
    async def test_run_max_cycles_enforced(self, mock_llm_client):
        settings = Settings(max_cycles=2)
        orch = Orchestrator(goal="test", client=mock_llm_client)

        cycle_call_count = 0

        async def mock_cycle():
            nonlocal cycle_call_count
            cycle_call_count += 1
            orch.tasks = [TaskModel(id="1", description="t", status="pending")]

        with patch.object(orch, "_cycle", mock_cycle):
            with patch("furrow.core.orchestrator.Settings", return_value=settings):
                await orch.run()

        assert orch.cycles == 2
        assert cycle_call_count == 2

    @pytest.mark.asyncio
    async def test_run_stops_when_done(self, mock_llm_client):
        settings = Settings(max_cycles=0)
        orch = Orchestrator(goal="test", client=mock_llm_client)

        cycle_call_count = 0

        async def mock_cycle():
            nonlocal cycle_call_count
            cycle_call_count += 1
            orch.tasks = [TaskModel(id="1", description="t", status="completed")]

        with patch.object(orch, "_cycle", mock_cycle):
            with patch("furrow.core.orchestrator.Settings", return_value=settings):
                await orch.run()

        assert cycle_call_count == 1

    @pytest.mark.asyncio
    async def test_run_increments_cycles(self, mock_llm_client):
        settings = Settings(max_cycles=3)
        orch = Orchestrator(goal="test", client=mock_llm_client)

        async def mock_cycle():
            orch.tasks = [TaskModel(id="1", description="t", status="pending")]

        with patch.object(orch, "_cycle", mock_cycle):
            with patch("furrow.core.orchestrator.Settings", return_value=settings):
                await orch.run()

        assert orch.cycles == 3

    @pytest.mark.asyncio
    async def test_cycle_updates_goal_on_test_failure(self, mock_llm_client, sample_plan):
        orch = Orchestrator(goal="original goal", client=mock_llm_client)

        with patch.object(orch.planner, "plan", return_value=sample_plan):
            with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
                mock_worker = MagicMock()
                mock_worker.run = AsyncMock(return_value="done")
                MockWorker.return_value = mock_worker

                with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                    mock_tester = MagicMock()
                    mock_tester.run = AsyncMock(
                        return_value=MagicMock(
                            passed=False,
                            summary="tests failed",
                            failures=["test_a failed", "test_b failed"],
                        )
                    )
                    MockTester.return_value = mock_tester

                    await orch._cycle()

        assert "Fix failing tests" in orch.goal
        assert "test_a failed" in orch.goal
        assert "test_b failed" in orch.goal
