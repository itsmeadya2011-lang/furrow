import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class TestOrchestratorInit:
    def test_init_with_goal(self):
        orchestrator = Orchestrator(goal="build app")
        assert orchestrator.goal == "build app"
        assert orchestrator.cycles == 0
        assert orchestrator._current_tasks == []
        assert orchestrator._last_test_passed is False

    def test_init_with_custom_client(self):
        mock_client = MagicMock()
        orchestrator = Orchestrator(goal="build app", client=mock_client)
        assert orchestrator.client is mock_client


class TestOrchestratorIsDone:
    @pytest.mark.asyncio
    async def test_all_tasks_completed_and_tests_passed(self):
        client = MagicMock()
        client.settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        orchestrator._last_test_passed = True

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_some_tasks_failed(self):
        client = MagicMock()
        client.settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ]
        orchestrator._last_test_passed = True

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_tests_failed(self):
        client = MagicMock()
        client.settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
        ]
        orchestrator._last_test_passed = False

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_no_tasks_planned(self):
        client = MagicMock()
        client.settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = []
        orchestrator._last_test_passed = True

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_no_tasks_not_done_before_tests(self):
        client = MagicMock()
        client.settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = []
        orchestrator._last_test_passed = False

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_max_cycles_reached(self):
        client = MagicMock()
        client.settings.max_cycles = 3
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator.cycles = 3
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="pending"),
        ]
        orchestrator._last_test_passed = False

        assert orchestrator._is_done() is True

    @pytest.mark.asyncio
    async def test_max_cycles_not_yet_reached(self):
        client = MagicMock()
        client.settings.max_cycles = 5
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator.cycles = 3
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
        ]
        orchestrator._last_test_passed = True

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_partial_completion_not_done(self):
        client = MagicMock()
        client.settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ]
        orchestrator._last_test_passed = True

        assert orchestrator._is_done() is False

    @pytest.mark.asyncio
    async def test_max_cycles_zero_means_no_limit(self):
        client = MagicMock()
        client.settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test", client=client)
        orchestrator.cycles = 100
        orchestrator._current_tasks = [
            TaskModel(id="1", description="a", status="completed"),
        ]
        orchestrator._last_test_passed = True

        assert orchestrator._is_done() is True


class TestOrchestratorGetTasks:
    @pytest.mark.asyncio
    async def test_get_tasks_returns_current_tasks(self):
        client = MagicMock()
        client.settings.max_cycles = 0
        orchestrator = Orchestrator(goal="test", client=client)
        tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
        orchestrator._current_tasks = tasks

        result = orchestrator._get_tasks()
        assert result is tasks
        assert len(result) == 2


class TestOrchestratorCycle:
    @pytest.mark.asyncio
    async def test_cycle_stores_tasks_and_test_results(self):
        mock_client = MagicMock()
        mock_client.settings.max_cycles = 0
        mock_client.settings.test_timeout = 120

        plan = Plan(
            tasks=[TaskModel(id="1", description="task 1", status="pending")],
            rationale="test plan",
        )

        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(return_value=plan)

        mock_tester = MagicMock()
        mock_tester.run = AsyncMock(return_value=TestResult(
            passed=True,
            summary="tests passed",
            failures=[],
        ))

        mock_worker = MagicMock()
        mock_worker.run = AsyncMock(return_value="done")

        with patch("furrow.core.orchestrator.PlannerAgent", return_value=mock_planner):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
                    with patch("furrow.core.orchestrator.console") as mock_console:
                        with patch("furrow.core.orchestrator.logger"):
                            with patch("furrow.core.orchestrator.Status"):
                                with patch("furrow.core.orchestrator.Panel"):
                                    with patch("furrow.core.orchestrator.Pretty"):
                                        orchestrator = Orchestrator(goal="test", client=mock_client)
                                        await orchestrator._cycle()

        assert len(orchestrator._current_tasks) == 1
        assert orchestrator._current_tasks[0].id == "1"
        assert orchestrator._current_tasks[0].status == "completed"
        assert orchestrator._last_test_passed is True

    @pytest.mark.asyncio
    async def test_cycle_with_no_tasks_sets_last_test_passed_true(self):
        mock_client = MagicMock()
        mock_client.settings.max_cycles = 0
        mock_client.settings.test_timeout = 120

        plan = Plan(tasks=[], rationale="no tasks")

        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(return_value=plan)

        with patch("furrow.core.orchestrator.PlannerAgent", return_value=mock_planner):
            with patch("furrow.core.orchestrator.console"):
                with patch("furrow.core.orchestrator.logger"):
                    with patch("furrow.core.orchestrator.Status"):
                        with patch("furrow.core.orchestrator.Panel"):
                            with patch("furrow.core.orchestrator.Pretty"):
                                orchestrator = Orchestrator(goal="test", client=mock_client)
                                await orchestrator._cycle()

        assert orchestrator._current_tasks == []
        assert orchestrator._last_test_passed is True

    @pytest.mark.asyncio
    async def test_cycle_updates_goal_on_test_failure(self):
        mock_client = MagicMock()
        mock_client.settings.max_cycles = 0
        mock_client.settings.test_timeout = 120

        plan = Plan(
            tasks=[TaskModel(id="1", description="task 1")],
            rationale="test",
        )

        mock_planner = MagicMock()
        mock_planner.plan = AsyncMock(return_value=plan)

        mock_tester = MagicMock()
        mock_tester.run = AsyncMock(return_value=TestResult(
            passed=False,
            summary="test failed",
            failures=["AssertionError in test_a", "TypeError in test_b"],
        ))

        mock_worker = MagicMock()
        mock_worker.run = AsyncMock(return_value="done")

        with patch("furrow.core.orchestrator.PlannerAgent", return_value=mock_planner):
            with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
                with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker):
                    with patch("furrow.core.orchestrator.console"):
                        with patch("furrow.core.orchestrator.logger"):
                            with patch("furrow.core.orchestrator.Status"):
                                with patch("furrow.core.orchestrator.Panel"):
                                    with patch("furrow.core.orchestrator.Pretty"):
                                        orchestrator = Orchestrator(goal="test", client=mock_client)
                                        await orchestrator._cycle()

        assert orchestrator._last_test_passed is False
        assert "Fix failing tests:" in orchestrator.goal
        assert "AssertionError in test_a" in orchestrator.goal
        assert "TypeError in test_b" in orchestrator.goal
