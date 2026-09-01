import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_client_with_settings(max_cycles: int = 0) -> MagicMock:
    """Build a MagicMock LLMClient with a settings attribute usable by Orchestrator."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.max_cycles = max_cycles
    client.settings.workspace = MagicMock()
    return client


class TestGetTasks:
    def test_returns_empty_list_when_no_plan(self):
        client = _make_client_with_settings()
        orch = Orchestrator(goal="do thing", client=client)
        assert orch.current_plan is None
        assert orch._get_tasks() == []

    def test_returns_tasks_when_plan_exists(self):
        client = _make_client_with_settings()
        orch = Orchestrator(goal="do thing", client=client)
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a"),
                TaskModel(id="2", description="b"),
            ],
            rationale="r",
        )
        tasks = orch._get_tasks()
        assert len(tasks) == 2
        assert [t.id for t in tasks] == ["1", "2"]


class TestIsDone:
    def test_max_cycles_limit_stops_run(self):
        client = _make_client_with_settings(max_cycles=3)
        orch = Orchestrator(goal="g", client=client)
        # Pretend a plan exists with all completed tasks; max_cycles should still trip first.
        orch.current_plan = Plan(
            tasks=[TaskModel(id="1", description="a", status="completed")],
            rationale="r",
        )
        orch.cycles = 3
        assert orch._is_done() is True

    def test_all_tasks_completed_returns_true(self):
        client = _make_client_with_settings(max_cycles=0)
        orch = Orchestrator(goal="g", client=client)
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="completed"),
            ],
            rationale="r",
        )
        orch.cycles = 1
        assert orch._is_done() is True

    def test_failed_tasks_returns_false(self):
        client = _make_client_with_settings(max_cycles=0)
        orch = Orchestrator(goal="g", client=client)
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="failed"),
            ],
            rationale="r",
        )
        orch.cycles = 1
        assert orch._is_done() is False

    def test_pending_tasks_returns_false(self):
        client = _make_client_with_settings(max_cycles=0)
        orch = Orchestrator(goal="g", client=client)
        orch.current_plan = Plan(
            tasks=[
                TaskModel(id="1", description="a", status="completed"),
                TaskModel(id="2", description="b", status="pending"),
            ],
            rationale="r",
        )
        orch.cycles = 1
        assert orch._is_done() is False


class TestCurrentGoalUpdateOnFailure:
    @pytest.mark.asyncio
    async def test_current_goal_changes_after_test_failure(self):
        client = _make_client_with_settings(max_cycles=0)
        orch = Orchestrator(goal="original goal", client=client)

        # Build a plan with one task
        plan = Plan(
            tasks=[TaskModel(id="1", description="task one")],
            rationale="r",
        )

        # Mock the planner to return our plan
        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=plan)

        # Mock the worker (constructed inside _cycle) to succeed
        worker_instance = MagicMock()
        worker_instance.run = AsyncMock(return_value='{"files": [], "summary": "ok"}')

        # Mock the tester to report failure with specific failure messages
        failure_msg = "AssertionError: expected 1 to equal 2"
        tester_instance = MagicMock()
        tester_instance.run = AsyncMock(
            return_value=TestResult(passed=False, summary="failed", failures=[failure_msg])
        )

        with patch("furrow.core.orchestrator.WorkerAgent", return_value=worker_instance), \
             patch("furrow.core.orchestrator.TesterAgent", return_value=tester_instance), \
             patch("furrow.core.orchestrator.console"):
            await orch._cycle()

        assert orch.current_goal != "original goal"
        assert failure_msg in orch.current_goal
        assert orch.current_goal.startswith("Fix failing tests:")

    @pytest.mark.asyncio
    async def test_current_goal_unchanged_when_tests_pass(self):
        client = _make_client_with_settings(max_cycles=0)
        orch = Orchestrator(goal="original goal", client=client)

        plan = Plan(
            tasks=[TaskModel(id="1", description="task one")],
            rationale="r",
        )

        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=plan)

        worker_instance = MagicMock()
        worker_instance.run = AsyncMock(return_value='{"files": [], "summary": "ok"}')

        tester_instance = MagicMock()
        tester_instance.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all good", failures=[])
        )

        with patch("furrow.core.orchestrator.WorkerAgent", return_value=worker_instance), \
             patch("furrow.core.orchestrator.TesterAgent", return_value=tester_instance), \
             patch("furrow.core.orchestrator.console"):
            await orch._cycle()

        assert orch.current_goal == "original goal"


class TestRunCycleFlow:
    @pytest.mark.asyncio
    async def test_run_terminates_when_done_after_one_cycle(self):
        client = _make_client_with_settings(max_cycles=0)
        orch = Orchestrator(goal="g", client=client)

        plan = Plan(
            tasks=[TaskModel(id="1", description="a")],
            rationale="r",
        )

        orch.planner = MagicMock()
        orch.planner.plan = AsyncMock(return_value=plan)

        worker_instance = MagicMock()
        worker_instance.run = AsyncMock(return_value='{"files": [], "summary": "ok"}')

        tester_instance = MagicMock()
        tester_instance.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )

        with patch("furrow.core.orchestrator.WorkerAgent", return_value=worker_instance), \
             patch("furrow.core.orchestrator.TesterAgent", return_value=tester_instance), \
             patch("furrow.core.orchestrator.console"):
            await orch.run()

        assert orch.cycles == 1
        # After first cycle, task should be marked completed
        assert orch.current_plan.tasks[0].status == "completed"