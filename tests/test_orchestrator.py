"""Tests for the Orchestrator."""
import json
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from tests.conftest import MockLLMClient


def _make_plan(tasks=None):
    if tasks is None:
        tasks = [TaskModel(id="1", description="Implement feature", files=["main.py"])]
    return Plan(tasks=tasks, rationale="test plan")


class TestOrchestratorInit:
    def test_stores_goal(self):
        orch = Orchestrator(goal="Build a castle")
        assert orch.goal == "Build a castle"

    def test_initializes_tasks_empty(self):
        orch = Orchestrator(goal="test")
        assert orch.tasks == []
        assert orch.cycles == 0

    def test_accepts_on_message(self):
        received = []
        async def on_msg(msg):
            received.append(msg)
        orch = Orchestrator(goal="test", on_message=on_msg)
        assert orch.on_message is on_msg


class TestOrchestratorIsDone:
    def test_empty_tasks_is_done(self):
        orch = Orchestrator(goal="test")
        orch.tasks = []
        assert orch._is_done() is True

    def test_all_completed_is_done(self):
        orch = Orchestrator(goal="test")
        orch.tasks = [
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        assert orch._is_done() is True

    def test_failed_tasks_not_done(self):
        orch = Orchestrator(goal="test")
        orch.tasks = [
            TaskModel(id="1", description="a", status="failed"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        assert orch._is_done() is False

    def test_pending_tasks_not_done(self):
        orch = Orchestrator(goal="test")
        orch.tasks = [
            TaskModel(id="1", description="a", status="pending"),
            TaskModel(id="2", description="b", status="completed"),
        ]
        assert orch._is_done() is False


class TestOrchestratorGetTasks:
    def test_returns_stored_tasks(self):
        orch = Orchestrator(goal="test")
        tasks = [TaskModel(id="1", description="a")]
        orch.tasks = tasks
        assert orch._get_tasks() is tasks


class TestOrchestratorMaxCycles:
    @pytest.mark.asyncio
    async def test_max_cycles_stops_loop(self):
        client = MockLLMClient(responses=[json.dumps({"tasks": [], "rationale": "nothing"})])
        client.settings.max_cycles = 1
        orch = Orchestrator(goal="done goal", client=client)

        # Mock the planner to return an empty plan (goal already complete)

        async def mock_plan(goal):
            return _make_plan(tasks=[])

        orch.planner.plan = mock_plan  # type: ignore
        await orch.run()
        assert orch.cycles == 1  # Should stop after 1 cycle


class TestOrchestratorRun:
    @pytest.mark.asyncio
    async def test_stores_tasks_after_cycle(self):
        """Tasks are stored on the orchestrator instance during _cycle()."""

        client = MockLLMClient()

        async def mock_plan(goal):
            return _make_plan()

        orch = Orchestrator(goal="build", client=client)
        orch.planner.plan = mock_plan  # type: ignore

        # Mock WorkerAgent and TesterAgent to avoid actual LLM calls
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            mock_worker_instance = AsyncMock()
            mock_worker_instance.run = AsyncMock(return_value="Worker result")
            MockWorker.return_value = mock_worker_instance

            mock_tester_instance = AsyncMock()
            mock_tester_instance.run = AsyncMock(return_value=TestResult(
                passed=True, summary="All good"
            ))
            MockTester.return_value = mock_tester_instance

            await orch._cycle()

        # Tasks should be stored
        assert len(orch.tasks) == 1
        assert orch.tasks[0].status == "completed"
        assert orch.tasks[0].result == "Worker result"

    @pytest.mark.asyncio
    async def test_on_message_receives_messages(self):
        received = []
        async def on_msg(msg):
            received.append(msg)

        client = MockLLMClient()

        async def mock_plan(goal):
            return _make_plan()

        orch = Orchestrator(goal="test goal", client=client, on_message=on_msg)
        orch.planner.plan = mock_plan  # type: ignore

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            mock_worker_instance = AsyncMock()
            mock_worker_instance.run = AsyncMock(return_value="done")
            MockWorker.return_value = mock_worker_instance

            mock_tester_instance = AsyncMock()
            mock_tester_instance.run = AsyncMock(return_value=TestResult(
                passed=True, summary="ok"
            ))
            MockTester.return_value = mock_tester_instance

            await orch._cycle()

        assert len(received) > 0

    @pytest.mark.asyncio
    async def test_failed_worker_marks_task_failed(self):
        client = MockLLMClient()

        async def mock_plan(goal):
            return _make_plan()

        orch = Orchestrator(goal="test", client=client)
        orch.planner.plan = mock_plan  # type: ignore

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            mock_worker_instance = AsyncMock()
            mock_worker_instance.run = AsyncMock(side_effect=RuntimeError("boom"))
            MockWorker.return_value = mock_worker_instance

            mock_tester_instance = AsyncMock()
            mock_tester_instance.run = AsyncMock(return_value=TestResult(
                passed=True, summary="ok"
            ))
            MockTester.return_value = mock_tester_instance

            await orch._cycle()

        assert orch.tasks[0].status == "failed"

    @pytest.mark.asyncio
    async def test_test_failure_updates_goal(self):
        client = MockLLMClient()

        async def mock_plan(goal):
            return _make_plan()

        orch = Orchestrator(goal="original goal", client=client)
        orch.planner.plan = mock_plan  # type: ignore

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            mock_worker_instance = AsyncMock()
            mock_worker_instance.run = AsyncMock(return_value="done")
            MockWorker.return_value = mock_worker_instance

            mock_tester_instance = AsyncMock()
            mock_tester_instance.run = AsyncMock(return_value=TestResult(
                passed=False,
                summary="Tests failed",
                failures=["test_foo: assertion error"],
            ))
            MockTester.return_value = mock_tester_instance

            await orch._cycle()

        assert "Fix failing tests" in orch.goal
        assert "test_foo" in orch.goal
