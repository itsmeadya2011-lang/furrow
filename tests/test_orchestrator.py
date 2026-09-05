"""Tests for the orchestrator."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class TestOrchestratorInit:
    def test_init_basic(self):
        orch = Orchestrator(goal="Add a login page")
        assert orch.goal == "Add a login page"
        assert orch.cycles == 0
        assert orch.tasks == []
        assert orch.tests_passed is False

    def test_init_with_client(self):
        mock_client = MagicMock(spec=LLMClient)
        orch = Orchestrator(goal="test", client=mock_client)
        assert orch.client is mock_client

    def test_init_with_max_cycles_override(self):
        orch = Orchestrator(goal="test", max_cycles=3)
        assert orch.max_cycles == 3


class TestOrchestratorIsDone:
    def test_no_tasks_done(self):
        orch = Orchestrator(goal="test")
        orch.tests_passed = True
        assert orch._is_done() is True

    def test_all_tasks_completed_and_tests_passed(self):
        orch = Orchestrator(goal="test")
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="completed"),
        ]
        orch.tests_passed = True
        assert orch._is_done() is True

    def test_tasks_completed_but_tests_failed(self):
        orch = Orchestrator(goal="test")
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
        ]
        orch.tests_passed = False
        assert orch._is_done() is False

    def test_task_failed(self):
        orch = Orchestrator(goal="test")
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="failed"),
        ]
        orch.tests_passed = True
        assert orch._is_done() is False

    def test_mixed_status(self):
        orch = Orchestrator(goal="test")
        orch.tasks = [
            TaskModel(id="1", description="task 1", status="completed"),
            TaskModel(id="2", description="task 2", status="pending"),
        ]
        orch.tests_passed = True
        assert orch._is_done() is False

    def test_get_tasks_returns_tracked_tasks(self):
        orch = Orchestrator(goal="test")
        task = TaskModel(id="1", description="task 1")
        orch.tasks = [task]
        assert orch._get_tasks() == [task]


class TestOrchestratorMaxCycles:
    @pytest.mark.asyncio
    async def test_max_cycles_zero_does_not_stop(self):
        """max_cycles=0 means unlimited; a no-op cycle should finish immediately."""
        orch = Orchestrator(goal="test", max_cycles=0)

        plan = Plan(tasks=[], rationale="Nothing to do")
        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            await orch._cycle()

        assert orch._is_done() is True


class TestOrchestratorCycle:
    @pytest.mark.asyncio
    async def test_cycle_with_no_tasks(self):
        orch = Orchestrator(goal="test")
        plan = Plan(tasks=[], rationale="Already done")

        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            await orch._cycle()

        assert orch.tests_passed is True  # set when no tasks
        assert orch.tasks == []

    @pytest.mark.asyncio
    async def test_cycle_with_completed_tasks_and_passing_tests(self):
        from furrow.agents.worker import WorkerAgent
        from furrow.agents.tester import TesterAgent

        orch = Orchestrator(goal="test")
        plan = Plan(tasks=[
            TaskModel(id="1", description="task 1"),
        ], rationale="plan")

        test_result = TestResult(passed=True, summary="All good")

        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="Done"):
                with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=test_result):
                    await orch._cycle()

        assert orch.tasks[0].status == "completed"
        assert orch.tests_passed is True
        assert orch.last_test_result is test_result

    @pytest.mark.asyncio
    async def test_cycle_with_failed_task(self):
        from furrow.agents.worker import WorkerAgent
        from furrow.agents.tester import TesterAgent

        orch = Orchestrator(goal="test")
        plan = Plan(tasks=[
            TaskModel(id="1", description="task 1"),
        ], rationale="plan")

        test_result = TestResult(passed=True, summary="OK")

        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock, side_effect=RuntimeError("boom")):
                with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=test_result):
                    await orch._cycle()

        assert orch.tasks[0].status == "failed"
        assert "boom" in orch.tasks[0].result

    @pytest.mark.asyncio
    async def test_cycle_with_failing_tests_updates_goal(self):
        from furrow.agents.worker import WorkerAgent
        from furrow.agents.tester import TesterAgent

        orch = Orchestrator(goal="Build a login page")
        plan = Plan(tasks=[
            TaskModel(id="1", description="task 1"),
        ], rationale="plan")

        test_result = TestResult(
            passed=False,
            summary="2 tests failed",
            failures=["Test login failed: invalid token", "Test logout failed: missing route"],
        )

        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="Done"):
                with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=test_result):
                    await orch._cycle()

        assert orch.tests_passed is False
        assert "Fix failing tests" in orch.goal
        assert "Test login failed" in orch.goal


class TestOrchestratorRun:
    @pytest.mark.asyncio
    async def test_run_completes_when_no_tasks(self):
        orch = Orchestrator(goal="test", max_cycles=5)
        plan = Plan(tasks=[], rationale="Already done")

        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            await orch.run()

        assert orch._is_done()
        assert orch.cycles == 1

    @pytest.mark.asyncio
    async def test_run_stops_on_max_cycles(self):
        orch = Orchestrator(goal="test", max_cycles=2)

        # Each cycle returns tasks that always complete but tests fail
        plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="plan")
        test_result = TestResult(passed=False, summary="fail", failures=["issue"])

        from furrow.agents.worker import WorkerAgent
        from furrow.agents.tester import TesterAgent

        with patch.object(orch.planner, "plan", new_callable=AsyncMock, return_value=plan):
            with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="Done"):
                with patch.object(TesterAgent, "run", new_callable=AsyncMock, return_value=test_result):
                    await orch.run()

        # Should have hit max_cycles=2 and stopped
        assert orch.cycles == 2
