"""Tests for the Orchestrator: task tracking, completion logic, and max_cycles."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_plan(tasks):
    return Plan(tasks=tasks, rationale="test plan")


def _make_test_result(passed=True, failures=None):
    return TestResult(
        passed=passed,
        summary="all passed" if passed else "failures detected",
        failures=failures or [],
    )


@pytest.fixture
def mock_client():
    """Create a mock LLMClient with settings attributes."""
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = 0
    client.settings.llm_timeout = 120
    client.settings.max_retries = 3
    client.complete = AsyncMock(return_value="{}")
    client.read_file = AsyncMock(return_value="<file content>")
    return client


class TestOrchestratorInit:
    def test_initial_state(self, mock_client):
        orch = Orchestrator(goal="test goal", client=mock_client)
        assert orch.goal == "test goal"
        assert orch.cycles == 0
        assert orch._tasks == []
        assert orch._test_results == []
        assert orch._on_message is None

    def test_default_client_created_when_none(self):
        """Orchestrator should create a default LLMClient when none is passed."""
        from furrow.llm import LLMClient
        orch = Orchestrator(goal="test")
        assert isinstance(orch.client, LLMClient)
        assert isinstance(orch.planner, type(orch.planner))  # planner is set


class TestIsDone:
    def test_returns_false_when_no_tasks(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        assert orch._is_done() is False

    def test_returns_true_when_all_completed_and_tests_pass(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        orch._test_results = [_make_test_result(passed=True)]
        assert orch._is_done() is True

    def test_returns_false_when_any_task_failed(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="failed"),
        ]
        orch._test_results = [_make_test_result(passed=True)]
        assert orch._is_done() is False

    def test_returns_false_when_all_completed_but_tests_failed(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="completed"),
        ]
        orch._test_results = [_make_test_result(passed=False, failures=["broken"])]
        assert orch._is_done() is False

    def test_returns_true_when_all_completed_no_test_results(self, mock_client):
        """If no test results exist but all tasks completed, consider done."""
        orch = Orchestrator(goal="test", client=mock_client)
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
        ]
        assert orch._is_done() is True

    def test_returns_false_when_some_tasks_incomplete(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch._tasks = [
            TaskModel(id="1", description="t1", status="completed"),
            TaskModel(id="2", description="t2", status="pending"),
        ]
        orch._test_results = [_make_test_result(passed=True)]
        assert orch._is_done() is False


class TestGetTasks:
    def test_returns_stored_tasks(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        tasks = [TaskModel(id="1", description="t1"), TaskModel(id="2", description="t2")]
        orch._tasks = tasks
        assert orch._get_tasks() == tasks

    def test_returns_empty_when_never_set(self, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        assert orch._get_tasks() == []


class TestEmitCallback:
    @pytest.mark.asyncio
    async def test_on_message_callback(self, mock_client):
        messages = []
        orch = Orchestrator(
            goal="test",
            client=mock_client,
            on_message=lambda msg: messages.append(msg),
        )
        await orch._emit("test message")
        assert "test message" in messages


class TestRunLoop:
    """Test the full run loop with mocked agents."""

    def _setup_orch(self, mock_client, plan_tasks=None, test_result=None, max_cycles=0):
        """Create an orchestrator with mocked planner/worker/tester agents."""
        orch = Orchestrator(goal="test", client=mock_client)
        mock_client.settings.max_cycles = max_cycles

        plan = Plan(tasks=plan_tasks or [], rationale="ok")
        orch.planner.plan = AsyncMock(return_value=plan)

        # WorkerAgent.run returns a string summary
        worker_results = [f"result for {t.id}" for t in (plan_tasks or [])]
        orch._worker_results = worker_results

        # Patch WorkerAgent.run by monkey-patching the class
        from furrow.agents.worker import WorkerAgent

        original_run = WorkerAgent.run

        async def mock_worker_run(self):
            idx = int(self.task.id) - 1 if self.task.id.isdigit() else 0
            return worker_results[min(idx, len(worker_results) - 1)] if worker_results else "ok"

        WorkerAgent.run = mock_worker_run

        # Patch TesterAgent.run
        from furrow.agents.tester import TesterAgent

        orch._test_result = test_result or _make_test_result(passed=True)

        async def mock_tester_run(self, goal, tasks):
            return orch._test_result

        TesterAgent.run = mock_tester_run

        orch._monkey_patch = (original_run, WorkerAgent, original_run)
        return orch

    @pytest.mark.asyncio
    async def test_run_completes_in_one_cycle(self, mock_client):
        tasks = [TaskModel(id="1", description="t1", files=["main.py"])]
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent
        original_worker = WorkerAgent.run
        original_tester = TesterAgent.run
        orch = self._setup_orch(mock_client, plan_tasks=tasks, test_result=_make_test_result(passed=True))

        try:
            await orch.run()
        finally:
            WorkerAgent.run = original_worker
            TesterAgent.run = original_tester

        assert orch.cycles == 1
        assert orch._tasks[0].status == "completed"

    @pytest.mark.asyncio
    async def test_run_continues_on_test_failure(self, mock_client):
        """When tests fail, the loop should continue (not exit immediately)."""
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent
        original_worker = WorkerAgent.run
        original_tester = TesterAgent.run
        tasks = [TaskModel(id="1", description="t1", files=["main.py"])]
        orch = self._setup_orch(mock_client, plan_tasks=tasks, test_result=_make_test_result(passed=False, failures=["bug"]))
        mock_client.settings.max_cycles = 5

        try:
            await orch.run()
        finally:
            WorkerAgent.run = original_worker
            TesterAgent.run = original_tester

        # Should stop because max_cycles was hit
        assert orch.cycles == 5

    @pytest.mark.asyncio
    async def test_run_stops_at_max_cycles(self, mock_client):
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent
        original_worker = WorkerAgent.run
        original_tester = TesterAgent.run
        tasks = [TaskModel(id="1", description="t1")]
        orch = self._setup_orch(mock_client, plan_tasks=tasks, test_result=_make_test_result(passed=False, failures=["x"]))
        mock_client.settings.max_cycles = 3

        try:
            await orch.run()
        finally:
            WorkerAgent.run = original_worker
            TesterAgent.run = original_tester

        assert orch.cycles == 3

    @pytest.mark.asyncio
    async def test_run_continues_when_task_fails(self, mock_client):
        """When a task fails (exception), the loop should continue in the next cycle."""
        from furrow.agents.tester import TesterAgent
        from furrow.agents.worker import WorkerAgent
        original_worker = WorkerAgent.run
        original_tester = TesterAgent.run
        tasks = [TaskModel(id="1", description="t1")]
        orch = self._setup_orch(mock_client, plan_tasks=tasks, test_result=_make_test_result(passed=True), max_cycles=2)

        # Override worker to raise an exception
        async def failing_worker_run(self):
            raise RuntimeError("worker crashed")

        WorkerAgent.run = failing_worker_run

        try:
            await orch.run()
        finally:
            WorkerAgent.run = original_worker
            TesterAgent.run = original_tester

        assert orch.cycles == 2
        assert orch._tasks[0].status == "failed"
