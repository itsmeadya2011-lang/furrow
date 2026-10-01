"""Tests for the Orchestrator."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TaskStatus, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def mock_client() -> MagicMock:
    """A mock LLMClient that returns parseable JSON for planning/testing."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.provider = "anthropic"
    client.settings.planner_model = "test-planner"
    client.settings.worker_model = "test-worker"
    client.settings.tester_model = "test-tester"
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = 10
    client.settings.max_tokens = 4096
    client.settings.timeout = 30.0
    client.settings.max_retries = 1
    client.settings.retry_delay = 0.01
    client.settings.workspace = MagicMock()
    client.settings.workspace.__truediv__ = lambda self, other: MagicMock()
    client.settings.goal = ""
    client.complete = AsyncMock(return_value="{}")
    client.write_file = AsyncMock()
    client.read_file = AsyncMock(return_value="")
    client.resolve_path = MagicMock(return_value="/tmp/test")
    return client


@pytest.fixture
def mock_planner(mock_client) -> MagicMock:
    planner = MagicMock()
    planner.plan = AsyncMock()
    return planner


@pytest.fixture
def mock_tester(mock_client) -> MagicMock:
    tester = MagicMock()
    tester.run = AsyncMock()
    return tester


class TestOrchestratorInit:
    def test_default_settings(self):
        orch = Orchestrator(goal="test goal")
        assert orch.goal == "test goal"
        assert orch.cycles == 0
        assert orch._all_tasks == []
        assert orch._last_test_result is None

    def test_custom_settings(self, mock_client):
        custom_settings = MagicMock()
        custom_settings.provider = "anthropic"
        custom_settings.max_parallel_tasks = 3
        custom_settings.max_cycles = 5
        custom_settings.max_tokens = 4096
        custom_settings.timeout = 30.0
        custom_settings.max_retries = 1
        custom_settings.retry_delay = 0.01
        custom_settings.workspace = MagicMock()
        custom_settings.workspace.__truediv__ = lambda self, other: MagicMock()
        custom_settings.goal = ""
        custom_settings.planner_model = "test"
        custom_settings.worker_model = "test"
        custom_settings.tester_model = "test"
        orch = Orchestrator(goal="test", client=mock_client, settings=custom_settings)
        assert orch.settings is custom_settings


class TestIsDone:
    def test_done_when_tests_pass(self):
        orch = Orchestrator(goal="test")
        orch.settings = MagicMock(max_cycles=10)
        orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])
        orch._cycle_had_tasks = True
        assert orch._is_done() is True

    def test_not_done_when_tests_fail(self):
        orch = Orchestrator(goal="test")
        orch._last_test_result = TestResult(passed=False, summary="bad", failures=["fail"])
        orch._cycle_had_tasks = True
        assert orch._is_done() is False

    def test_not_done_when_no_test_result(self):
        orch = Orchestrator(goal="test")
        orch.settings = MagicMock(max_cycles=10)
        orch._cycle_had_tasks = True
        assert orch._is_done() is False

    def test_done_when_max_cycles_reached(self):
        orch = Orchestrator(goal="test")
        orch.cycles = 5
        orch.settings = MagicMock(max_cycles=5)
        orch._cycle_had_tasks = True
        assert orch._is_done() is True

    def test_not_done_when_max_cycles_not_reached(self):
        orch = Orchestrator(goal="test")
        orch.cycles = 3
        orch.settings = MagicMock(max_cycles=10)
        orch._cycle_had_tasks = True
        assert orch._is_done() is False

    def test_max_cycles_zero_means_unlimited(self):
        orch = Orchestrator(goal="test")
        orch.cycles = 100
        orch.settings = MagicMock(max_cycles=0)
        orch._cycle_had_tasks = True
        assert orch._is_done() is False

    def test_done_when_no_tasks_and_no_test_result(self):
        """Stuck: empty plan and no test result means nothing can be done."""
        orch = Orchestrator(goal="test")
        orch.settings = MagicMock(max_cycles=10)
        orch._cycle_had_tasks = False
        orch._last_test_result = None
        assert orch._is_done() is True


class TestCycle:
    @pytest.mark.asyncio
    async def test_cycle_with_passing_tests(self, mock_client, mock_planner, mock_tester):
        orch = Orchestrator(goal="make it work", client=mock_client)
        orch.planner = mock_planner
        orch.tester = mock_tester
        orch._emit = AsyncMock()

        task = TaskModel(id="1", description="do thing", files=["main.py"])
        plan = Plan(tasks=[task], rationale="ok")
        mock_planner.plan.return_value = plan
        mock_tester.run.return_value = TestResult(passed=True, summary="all good", failures=[])

        await orch._cycle()

        assert mock_planner.plan.called
        assert mock_tester.run.called
        assert orch._last_test_result.passed is True
        # Task should be tracked and completed
        assert len(orch._all_tasks) == 1
        assert orch._all_tasks[0].status == TaskStatus.COMPLETED
        assert orch._cycle_had_tasks is True

    @pytest.mark.asyncio
    async def test_cycle_with_failing_tasks(self, mock_client, mock_planner, mock_tester):
        orch = Orchestrator(goal="make it work", client=mock_client)
        orch.planner = mock_planner
        orch.tester = mock_tester
        orch._emit = AsyncMock()

        plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        mock_planner.plan.return_value = plan

        # Patch WorkerAgent.run to raise
        with patch("furrow.core.orchestrator.WorkerAgent") as mock_worker_cls:
            mock_worker = MagicMock()
            mock_worker.run = AsyncMock(side_effect=RuntimeError("boom"))
            mock_worker_cls.return_value = mock_worker
            mock_tester.run.return_value = TestResult(
                passed=False, summary="tests fail", failures=["f1"]
            )

            await orch._cycle()

        assert orch._all_tasks[0].status == TaskStatus.FAILED
        assert "boom" in orch._all_tasks[0].result

    @pytest.mark.asyncio
    async def test_cycle_updates_goal_on_test_failure(self, mock_client, mock_planner, mock_tester):
        orch = Orchestrator(goal="original goal", client=mock_client)
        orch.planner = mock_planner
        orch.tester = mock_tester
        orch._emit = AsyncMock()

        plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
        mock_planner.plan.return_value = plan
        mock_tester.run.return_value = TestResult(
            passed=False, summary="tests fail", failures=["test_auth failed"],
        )

        original_goal = orch.goal
        await orch._cycle()

        # Goal should be updated to include fix instructions
        assert orch.goal != original_goal
        assert "Fix failing tests" in orch.goal
        assert "test_auth failed" in orch.goal

    @pytest.mark.asyncio
    async def test_no_tasks_planned(self, mock_client, mock_planner, mock_tester):
        orch = Orchestrator(goal="done", client=mock_client)
        orch.planner = mock_planner
        orch.tester = mock_tester
        orch._emit = AsyncMock()

        plan = Plan(tasks=[], rationale="already complete")
        mock_planner.plan.return_value = plan

        await orch._cycle()

        assert len(orch._all_tasks) == 0
        # Tester should NOT be called when there are no tasks
        assert mock_tester.run.called is False
        assert orch._cycle_had_tasks is False


class TestRunLoop:
    def _make_task_plan(self) -> Plan:
        return Plan(
            tasks=[TaskModel(id="1", description="do thing", files=["main.py"])],
            rationale="ok",
        )

    @pytest.mark.asyncio
    async def test_run_stops_when_tests_pass(self, mock_client, mock_planner, mock_tester):
        orch = Orchestrator(goal="test", client=mock_client)
        orch.planner = mock_planner
        orch.tester = mock_tester
        orch._emit = AsyncMock()

        mock_planner.plan.return_value = self._make_task_plan()
        mock_tester.run.return_value = TestResult(passed=True, summary="ok", failures=[])

        await orch.run()

        assert orch.cycles == 1

    @pytest.mark.asyncio
    async def test_run_respects_max_cycles(self, mock_client, mock_planner, mock_tester):
        mock_client.settings.max_cycles = 3
        orch = Orchestrator(goal="test", client=mock_client)
        orch.planner = mock_planner
        orch.tester = mock_tester
        orch._emit = AsyncMock()

        mock_planner.plan.return_value = self._make_task_plan()
        mock_tester.run.return_value = TestResult(passed=False, summary="fail", failures=["fail"])

        await orch.run()

        assert orch.cycles == 3

    @pytest.mark.asyncio
    async def test_run_still_running_when_tests_fail(self, mock_client, mock_planner, mock_tester):
        mock_client.settings.max_cycles = 10
        orch = Orchestrator(goal="test", client=mock_client)
        orch.planner = mock_planner
        orch.tester = mock_tester
        orch._emit = AsyncMock()

        mock_planner.plan.return_value = self._make_task_plan()
        mock_tester.run.return_value = TestResult(passed=False, summary="fail", failures=["fail"])

        await orch.run()

        assert orch.cycles == 10
        assert orch._last_test_result.passed is False

    @pytest.mark.asyncio
    async def test_run_stops_on_empty_plan(self, mock_client, mock_planner, mock_tester):
        """Empty plan means no progress possible → stop after 1 cycle."""
        orch = Orchestrator(goal="done", client=mock_client)
        orch.planner = mock_planner
        orch.tester = mock_tester
        orch._emit = AsyncMock()

        plan = Plan(tasks=[], rationale="nothing to do")
        mock_planner.plan.return_value = plan

        await orch.run()

        assert orch.cycles == 1
        assert mock_tester.run.called is False
