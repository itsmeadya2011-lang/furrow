import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


class TestOrchestrator:
    """Tests for the Orchestrator class."""

    def test_orchestrator_init_with_mock_client(self):
        """Test orchestrator initialization with mock LLMClient."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        assert orchestrator.goal == "test goal"
        assert orchestrator.client is mock_client
        assert orchestrator.cycles == 0
        assert orchestrator._current_plan is None

    def test_orchestrator_init_creates_default_client(self):
        """Test orchestrator creates default LLMClient when none provided."""
        with patch("furrow.core.orchestrator.LLMClient") as mock_llm:
            mock_llm.return_value = MagicMock(spec=LLMClient)
            orchestrator = Orchestrator(goal="test goal")
            assert orchestrator.goal == "test goal"
            assert orchestrator.client is not None

    def test_is_done_returns_false_when_tasks_pending(self):
        """Test _is_done() returns False when tasks are pending."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator._current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="pending"),
            ],
            rationale="test plan",
        )
        assert orchestrator._is_done() is False

    def test_is_done_returns_true_when_all_tasks_completed(self):
        """Test _is_done() returns True when all tasks completed."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator._current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="completed"),
            ],
            rationale="test plan",
        )
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_any_task_failed(self):
        """Test _is_done() returns False when any task failed."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator._current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="failed"),
            ],
            rationale="test plan",
        )
        assert orchestrator._is_done() is False

    def test_is_done_returns_true_when_no_tasks(self):
        """Test _is_done() returns True when there are no tasks."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        orchestrator._current_plan = Plan(tasks=[], rationale="empty plan")
        assert orchestrator._is_done() is True

    def test_is_done_returns_false_when_no_plan(self):
        """Test _is_done() returns True when no plan exists (empty task list)."""
        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)
        # No plan set, _get_tasks returns empty list
        # completed (0) >= len(tasks) (0) is True
        assert orchestrator._is_done() is True

    @patch("furrow.core.orchestrator.Settings")
    @patch("furrow.core.orchestrator.WorkerAgent")
    @patch("furrow.core.orchestrator.TesterAgent")
    @patch("furrow.core.orchestrator.PlannerAgent")
    def test_max_cycles_limits_execution(
        self, mock_tester_cls, mock_worker_cls, mock_planner_cls, mock_settings_cls
    ):
        """Test max_cycles limits execution."""
        mock_settings = MagicMock()
        mock_settings.max_cycles = 2
        mock_settings_cls.return_value = mock_settings

        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(
            return_value=Plan(
                tasks=[TaskModel(id="1", description="task 1", status="completed")],
                rationale="test plan",
            )
        )
        mock_planner_cls.return_value = mock_planner

        mock_worker = MagicMock()
        mock_worker.run = AsyncMock(return_value="done")
        mock_worker_cls.return_value = mock_worker

        mock_tester = MagicMock()
        mock_tester.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all good", failures=[])
        )
        mock_tester_cls.return_value = mock_tester

        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)

        asyncio.run(orchestrator.run())

        # Should stop after max_cycles (2) iterations
        assert orchestrator.cycles == 2

    @patch("furrow.core.orchestrator.Settings")
    @patch("furrow.core.orchestrator.PlannerAgent")
    def test_cycle_stores_plan_for_get_tasks(self, mock_planner_cls, mock_settings_cls):
        """Test _cycle() stores plan for _get_tasks()."""
        mock_settings = MagicMock()
        mock_settings.max_cycles = 1
        mock_settings_cls.return_value = mock_settings

        expected_plan = Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="test plan",
        )
        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(return_value=expected_plan)
        mock_planner_cls.return_value = mock_planner

        mock_client = MagicMock(spec=LLMClient)
        orchestrator = Orchestrator(goal="test goal", client=mock_client)

        # Before cycle, no plan
        assert orchestrator._current_plan is None
        assert orchestrator._get_tasks() == []

        # Run one cycle
        asyncio.run(orchestrator._cycle())

        # After cycle, plan is stored
        assert orchestrator._current_plan is expected_plan
        assert len(orchestrator._get_tasks()) == 1
        assert orchestrator._get_tasks()[0].id == "1"
