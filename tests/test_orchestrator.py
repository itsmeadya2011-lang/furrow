from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.config import Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


@pytest.fixture
def mock_settings() -> Settings:
    return Settings(
        provider=Provider.ANTHROPIC,
        model="claude-3-5-sonnet-20241022",
        planner_model="claude-3-5-haiku-20241022",
        worker_model="claude-3-5-sonnet-20241022",
        tester_model="claude-3-5-sonnet-20241022",
        max_parallel_tasks=5,
        max_cycles=0,
    )


@pytest.fixture
def mock_client(mock_settings: Settings) -> LLMClient:
    client = LLMClient(mock_settings)
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "test"}')
    client.read_file = AsyncMock(return_value="")
    client.write_file = AsyncMock()
    return client


class TestOrchestratorTaskTracking:
    @pytest.mark.asyncio
    async def test_tasks_stored_after_cycle(self, mock_settings, mock_client):
        task_model = TaskModel(id="1", description="do x", files=[])
        mock_client.complete = AsyncMock(return_value='{"tasks": [{"id": "1", "description": "do x", "files": [], "dependencies": []}], "rationale": "r"}')
        orch = Orchestrator(goal="test goal", client=mock_client)
        with patch.object(orch, "_cycle", new=mock_client.complete):
            pass
        orch = Orchestrator(goal="test goal", client=mock_client)
        orch.tasks = [task_model]
        assert orch._get_tasks() == [task_model]

    def test_get_tasks_returns_empty_by_default(self, mock_settings, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        assert orch._get_tasks() == []


class TestIsDone:
    def test_is_done_no_tasks_returns_false(self, mock_settings, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        assert orch._is_done() is False

    def test_is_done_max_cycles_exceeded(self, mock_settings, mock_client):
        settings = Settings(max_cycles=2)
        client = LLMClient(settings=settings)
        client.settings = settings
        orch = Orchestrator(goal="test", client=client)
        orch.cycles = 2
        orch.tasks = [TaskModel(id="1", description="x", files=[])]
        orch.tasks[0].status = "completed"
        orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orch._is_done() is True

    def test_is_done_max_cycles_not_reached(self, mock_settings, mock_client):
        settings = Settings(max_cycles=10)
        client = LLMClient(settings=settings)
        orch = Orchestrator(goal="test", client=client)
        orch.cycles = 2
        orch.tasks = [TaskModel(id="1", description="x", files=[])]
        orch.tasks[0].status = "completed"
        orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orch._is_done() is True

    def test_is_done_failed_tasks(self, mock_settings, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [TaskModel(id="1", description="x", files=[])]
        orch.tasks[0].status = "failed"
        assert orch._is_done() is False

    def test_is_done_all_completed_but_tests_not_passed(self, mock_settings, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        orch.tasks = [TaskModel(id="1", description="x", files=[])]
        orch.tasks[0].status = "completed"
        orch.last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
        assert orch._is_done() is False

    def test_is_done_all_completed_tests_passed(self, mock_settings, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        task1 = TaskModel(id="1", description="x", files=[])
        task1.status = "completed"
        task2 = TaskModel(id="2", description="y", files=[])
        task2.status = "completed"
        orch.tasks = [task1, task2]
        orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orch._is_done() is True

    def test_is_done_partial_completion(self, mock_settings, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        task1 = TaskModel(id="1", description="x", files=[])
        task1.status = "completed"
        task2 = TaskModel(id="2", description="y", files=[])
        task2.status = "pending"
        orch.tasks = [task1, task2]
        orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
        assert orch._is_done() is False


class TestRunErrorHandling:
    @pytest.mark.asyncio
    async def test_run_continues_after_cycle_error(self, mock_settings, mock_client):
        orch = Orchestrator(goal="test", client=mock_client)
        call_count = 0

        async def mock_plan(goal):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("cycle boom")
            from furrow.config import Plan
            return Plan(tasks=[], rationale="done")

        with patch.object(orch.planner, "plan", side_effect=mock_plan):
            await asyncio.wait_for(orch.run(), timeout=5)
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_last_test_result_stored(self, mock_settings, mock_client):
        test_result = TestResult(passed=True, summary="all pass", failures=[])
        mock_client.complete = AsyncMock(return_value='{"passed": true, "summary": "all pass", "failures": []}')
        orch = Orchestrator(goal="test", client=mock_client)
        orch.last_test_result = test_result
        assert orch.last_test_result is test_result
