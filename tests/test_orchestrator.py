from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestOrchestrator:
    def test_is_done_no_tasks(self) -> None:
        """_is_done() returns True when there are no tasks."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator.tasks = []

        assert orchestrator._is_done() is True

    def test_is_done_all_completed(self) -> None:
        """_is_done() returns True when all tasks are completed and tests passed."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        orchestrator.test_result = TestResult(passed=True, summary="All good")

        assert orchestrator._is_done() is True

    def test_is_done_has_failed(self) -> None:
        """_is_done() returns False when a task failed."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="failed"),
        ]
        orchestrator.test_result = TestResult(passed=True, summary="All good")

        assert orchestrator._is_done() is False

    def test_is_done_tests_failed(self) -> None:
        """_is_done() returns False when tasks completed but tests failed."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        orchestrator.test_result = TestResult(passed=False, summary="Tests failed")

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_max_cycles_limits_run(self) -> None:
        """With max_cycles=1, run() completes after 1 cycle."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)

        # Mock the planner to return a plan with tasks
        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[TaskModel(id="1", description="task 1")],
                rationale="test",
            )
        )
        orchestrator.planner = mock_planner

        # Mock _cycle to avoid actual execution
        orchestrator._cycle = AsyncMock()
        orchestrator._is_done = MagicMock(return_value=True)

        # Set max_cycles to 1
        with patch("furrow.core.orchestrator.settings") as mock_settings:
            mock_settings.max_cycles = 1
            await orchestrator.run()

        # Should have incremented cycles once and called _cycle once
        assert orchestrator.cycles == 1
        orchestrator._cycle.assert_called_once()

    @pytest.mark.asyncio
    async def test_tasks_stored_after_cycle(self) -> None:
        """After _cycle(), orchestrator.tasks is populated."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)

        # Create test tasks
        task1 = TaskModel(id="1", description="task 1")
        task2 = TaskModel(id="2", description="task 2")

        # Mock the planner to return a plan with tasks
        mock_planner = MagicMock(spec=PlannerAgent)
        mock_planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[task1, task2],
                rationale="test",
            )
        )
        orchestrator.planner = mock_planner

        # Mock WorkerAgent.run to return results
        async def mock_worker_run():
            return "result"

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorkerAgent:
            mock_worker_instance = MagicMock()
            mock_worker_instance.run = AsyncMock(return_value="result")
            MockWorkerAgent.return_value = mock_worker_instance

            # Mock TesterAgent.run to return passing result
            with patch("furrow.core.orchestrator.TesterAgent") as MockTesterAgent:
                mock_tester_instance = MagicMock()
                mock_tester_instance.run = AsyncMock(
                    return_value=TestResult(passed=True, summary="All good")
                )
                MockTesterAgent.return_value = mock_tester_instance

                await orchestrator._cycle()

        assert len(orchestrator.tasks) == 2
        assert orchestrator.tasks[0].id == "1"
        assert orchestrator.tasks[1].id == "2"
