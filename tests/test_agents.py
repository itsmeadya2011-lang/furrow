"""Tests for the agent classes."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, TaskModel, TestResult
from furrow.llm import LLMClient


class TestPlannerAgent:
    @pytest.mark.asyncio
    async def test_plan_success(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = MagicMock(planner_model="test-model")
        mock_client.complete = AsyncMock(
            return_value='{"tasks": [{"id": "1", "description": "test task", "files": [], "dependencies": []}], "rationale": "ok"}'
        )

        agent = PlannerAgent(client=mock_client)
        plan = await agent.plan("Test goal")

        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "test task"

    @pytest.mark.asyncio
    async def test_plan_invalid_json(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = MagicMock(planner_model="test-model")
        mock_client.complete = AsyncMock(return_value="not valid json")

        agent = PlannerAgent(client=mock_client)

        with pytest.raises(ValueError, match="Failed to parse plan"):
            await agent.plan("Test goal")


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_success(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = MagicMock(worker_model="test-model")
        mock_client.complete = AsyncMock(return_value="Task completed successfully")

        task = TaskModel(id="1", description="Implement feature", files=["src/feature.py"])
        agent = WorkerAgent(task=task, client=mock_client)
        result = await agent.run()

        assert result == "Task completed successfully"
        mock_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_with_empty_files(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = MagicMock(worker_model="test-model")
        mock_client.complete = AsyncMock(return_value="Done")

        task = TaskModel(id="1", description="Simple task")
        agent = WorkerAgent(task=task, client=mock_client)
        result = await agent.run()

        assert result == "Done"


class TestTesterAgent:
    @pytest.mark.asyncio
    async def test_run_tests_passed(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = MagicMock(tester_model="test-model")
        mock_client.complete = AsyncMock(
            return_value='{"passed": true, "summary": "All tests passed", "failures": []}'
        )

        agent = TesterAgent(client=mock_client)
        result = await agent.run("Test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_run_tests_failed(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = MagicMock(tester_model="test-model")
        mock_client.complete = AsyncMock(
            return_value='{"passed": false, "summary": "2 tests failed", "failures": ["test_a", "test_b"]}'
        )

        agent = TesterAgent(client=mock_client)
        result = await agent.run("Test goal", [])

        assert result.passed is False
        assert len(result.failures) == 2

    @pytest.mark.asyncio
    async def test_run_invalid_json_response(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = MagicMock(tester_model="test-model")
        mock_client.complete = AsyncMock(return_value="not json at all")

        agent = TesterAgent(client=mock_client)
        result = await agent.run("Test goal", [])

        assert result.passed is False
        assert "Failed to parse" in result.summary

    @pytest.mark.asyncio
    async def test_run_tests_subprocess_error(self):
        mock_client = MagicMock(spec=LLMClient)
        mock_client.settings = MagicMock(tester_model="test-model")
        mock_client.complete = AsyncMock(
            return_value='{"passed": true, "summary": "No tests found", "failures": []}'
        )

        agent = TesterAgent(client=mock_client)

        import asyncio
        from unittest.mock import patch

        with patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("No pytest")
            result = await agent.run("Test goal", [])

        assert isinstance(result, TestResult)