from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.llm import LLMClient


class TestPlannerAgent:
    def test_init_default(self):
        agent = PlannerAgent()
        assert agent.client is not None

    def test_init_with_client(self, mock_llm_client):
        agent = PlannerAgent(client=mock_llm_client)
        assert agent.client == mock_llm_client

    @pytest.mark.asyncio
    async def test_plan_returns_plan(self, mock_llm_client):
        plan_json = json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "Task 1", "files": ["a.py"]},
                    {"id": "2", "description": "Task 2", "files": ["b.py"]},
                ],
                "rationale": "Test rationale",
            }
        )
        mock_llm_client.complete = AsyncMock(return_value=plan_json)

        agent = PlannerAgent(client=mock_llm_client)
        result = await agent.plan("test goal")

        assert isinstance(result, Plan)
        assert len(result.tasks) == 2
        assert result.tasks[0].id == "1"
        assert result.tasks[0].description == "Task 1"
        assert result.rationale == "Test rationale"

    @pytest.mark.asyncio
    async def test_plan_calls_llm_with_correct_model(self, mock_llm_client):
        plan_json = json.dumps({"tasks": [], "rationale": "empty"})
        mock_llm_client.complete = AsyncMock(return_value=plan_json)
        mock_llm_client.settings.planner_model = "test-planner-model"

        agent = PlannerAgent(client=mock_llm_client)
        await agent.plan("test goal")

        mock_llm_client.complete.assert_called_once()
        call_kwargs = mock_llm_client.complete.call_args[1]
        assert call_kwargs["model"] == "test-planner-model"

    @pytest.mark.asyncio
    async def test_plan_includes_goal_in_prompt(self, mock_llm_client):
        plan_json = json.dumps({"tasks": [], "rationale": "empty"})
        mock_llm_client.complete = AsyncMock(return_value=plan_json)

        agent = PlannerAgent(client=mock_llm_client)
        await agent.plan("my specific goal")

        call_args = mock_llm_client.complete.call_args[0]
        prompt = call_args[0]
        assert "my specific goal" in prompt

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_json(self, mock_llm_client):
        mock_llm_client.complete = AsyncMock(return_value="not valid json")

        agent = PlannerAgent(client=mock_llm_client)
        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await agent.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_raises_on_invalid_plan_structure(self, mock_llm_client):
        invalid_json = json.dumps({"invalid_key": "no tasks key"})
        mock_llm_client.complete = AsyncMock(return_value=invalid_json)

        agent = PlannerAgent(client=mock_llm_client)
        with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
            await agent.plan("test goal")

    @pytest.mark.asyncio
    async def test_plan_with_empty_tasks(self, mock_llm_client):
        plan_json = json.dumps({"tasks": [], "rationale": "nothing to do"})
        mock_llm_client.complete = AsyncMock(return_value=plan_json)

        agent = PlannerAgent(client=mock_llm_client)
        result = await agent.plan("test goal")

        assert isinstance(result, Plan)
        assert result.tasks == []
        assert result.rationale == "nothing to do"


class TestWorkerAgent:
    def test_init(self, sample_task):
        agent = WorkerAgent(task=sample_task)
        assert agent.task == sample_task
        assert agent.client is not None

    def test_init_with_client(self, sample_task, mock_llm_client):
        agent = WorkerAgent(task=sample_task, client=mock_llm_client)
        assert agent.client == mock_llm_client

    @pytest.mark.asyncio
    async def test_run_returns_result(self, sample_task, mock_llm_client):
        mock_llm_client.complete = AsyncMock(return_value="task completed")

        agent = WorkerAgent(task=sample_task, client=mock_llm_client)
        result = await agent.run()

        assert result == "task completed"

    @pytest.mark.asyncio
    async def test_run_calls_llm(self, sample_task, mock_llm_client):
        mock_llm_client.complete = AsyncMock(return_value="done")

        agent = WorkerAgent(task=sample_task, client=mock_llm_client)
        await agent.run()

        mock_llm_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_uses_worker_model(self, sample_task, mock_llm_client):
        mock_llm_client.complete = AsyncMock(return_value="done")
        mock_llm_client.settings.worker_model = "test-worker-model"

        agent = WorkerAgent(task=sample_task, client=mock_llm_client)
        await agent.run()

        call_kwargs = mock_llm_client.complete.call_args[1]
        assert call_kwargs["model"] == "test-worker-model"

    @pytest.mark.asyncio
    async def test_run_includes_task_description_in_prompt(self, mock_llm_client):
        task = TaskModel(id="1", description="Implement auth system", files=[])
        mock_llm_client.complete = AsyncMock(return_value="done")

        agent = WorkerAgent(task=task, client=mock_llm_client)
        await agent.run()

        call_args = mock_llm_client.complete.call_args[0]
        prompt = call_args[0]
        assert "Implement auth system" in prompt

    @pytest.mark.asyncio
    async def test_run_includes_files_in_prompt(self, mock_llm_client):
        task = TaskModel(id="1", description="Task", files=["src/auth.py", "tests/test_auth.py"])
        mock_llm_client.complete = AsyncMock(return_value="done")

        agent = WorkerAgent(task=task, client=mock_llm_client)
        await agent.run()

        call_args = mock_llm_client.complete.call_args[0]
        prompt = call_args[0]
        assert "src/auth.py" in prompt
        assert "tests/test_auth.py" in prompt

    @pytest.mark.asyncio
    async def test_run_with_no_files(self, mock_llm_client):
        task = TaskModel(id="1", description="Task", files=[])
        mock_llm_client.complete = AsyncMock(return_value="done")

        agent = WorkerAgent(task=task, client=mock_llm_client)
        await agent.run()

        call_args = mock_llm_client.complete.call_args[0]
        prompt = call_args[0]
        assert "any" in prompt


class TestTesterAgent:
    def test_init_default(self):
        agent = TesterAgent()
        assert agent.client is not None

    def test_init_with_client(self, mock_llm_client):
        agent = TesterAgent(client=mock_llm_client)
        assert agent.client == mock_llm_client

    @pytest.mark.asyncio
    async def test_run_returns_test_result(self, mock_llm_client):
        test_json = json.dumps(
            {"passed": True, "summary": "All tests passed", "failures": []}
        )
        mock_llm_client.complete = AsyncMock(return_value=test_json)

        agent = TesterAgent(client=mock_llm_client)
        result = await agent.run("test goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "All tests passed"
        assert result.failures == []

    @pytest.mark.asyncio
    async def test_run_with_failed_tests(self, mock_llm_client):
        test_json = json.dumps(
            {
                "passed": False,
                "summary": "2 tests failed",
                "failures": ["test_a", "test_b"],
            }
        )
        mock_llm_client.complete = AsyncMock(return_value=test_json)

        agent = TesterAgent(client=mock_llm_client)
        result = await agent.run("test goal", [])

        assert result.passed is False
        assert result.summary == "2 tests failed"
        assert result.failures == ["test_a", "test_b"]

    @pytest.mark.asyncio
    async def test_run_uses_tester_model(self, mock_llm_client):
        test_json = json.dumps({"passed": True, "summary": "ok", "failures": []})
        mock_llm_client.complete = AsyncMock(return_value=test_json)
        mock_llm_client.settings.tester_model = "test-tester-model"

        agent = TesterAgent(client=mock_llm_client)
        await agent.run("goal", [])

        call_kwargs = mock_llm_client.complete.call_args[1]
        assert call_kwargs["model"] == "test-tester-model"

    @pytest.mark.asyncio
    async def test_run_handles_non_json_response(self, mock_llm_client):
        mock_llm_client.complete = AsyncMock(return_value="Tests passed successfully")

        agent = TesterAgent(client=mock_llm_client)
        result = await agent.run("goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is True
        assert result.summary == "Tests passed successfully"

    @pytest.mark.asyncio
    async def test_run_handles_non_json_failure_response(self, mock_llm_client):
        mock_llm_client.complete = AsyncMock(return_value="Tests failed with errors")

        agent = TesterAgent(client=mock_llm_client)
        result = await agent.run("goal", [])

        assert isinstance(result, TestResult)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_run_handles_test_execution_error(self, mock_llm_client):
        mock_llm_client.complete = AsyncMock(
            return_value=json.dumps({"passed": True, "summary": "ok", "failures": []})
        )

        agent = TesterAgent(client=mock_llm_client)

        with patch.object(
            agent, "_run_tests", side_effect=Exception("test runner crashed")
        ):
            result = await agent.run("goal", [])

        assert result.passed is False
        assert "test runner crashed" in result.summary

    @pytest.mark.asyncio
    async def test_run_includes_goal_in_prompt(self, mock_llm_client):
        test_json = json.dumps({"passed": True, "summary": "ok", "failures": []})
        mock_llm_client.complete = AsyncMock(return_value=test_json)

        agent = TesterAgent(client=mock_llm_client)
        await agent.run("my specific goal", [])

        call_args = mock_llm_client.complete.call_args[0]
        prompt = call_args[0]
        assert "my specific goal" in prompt

    @pytest.mark.asyncio
    async def test_run_includes_test_output_in_prompt(self, mock_llm_client):
        test_json = json.dumps({"passed": True, "summary": "ok", "failures": []})
        mock_llm_client.complete = AsyncMock(return_value=test_json)

        agent = TesterAgent(client=mock_llm_client)

        with patch.object(agent, "_run_tests", return_value="collected 5 tests"):
            await agent.run("goal", [])

        call_args = mock_llm_client.complete.call_args[0]
        prompt = call_args[0]
        assert "collected 5 tests" in prompt

    @pytest.mark.asyncio
    async def test_run_tests_no_runner_found(self, mock_llm_client):
        agent = TesterAgent(client=mock_llm_client)

        with patch("furrow.agents.tester.asyncio.create_subprocess_exec") as mock_exec:
            mock_exec.side_effect = FileNotFoundError("not found")
            result = await agent._run_tests()

        assert result == "No test runner found."

    @pytest.mark.asyncio
    async def test_run_tests_timeout(self, mock_llm_client):
        import asyncio

        agent = TesterAgent(client=mock_llm_client)

        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_proc.kill = MagicMock()

        with patch(
            "furrow.agents.tester.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with patch(
                "furrow.agents.tester.asyncio.wait_for", side_effect=asyncio.TimeoutError()
            ):
                result = await agent._run_tests()

        assert "timed out" in result.lower() or result == "No test runner found."

    @pytest.mark.asyncio
    async def test_run_tests_successful_execution(self, mock_llm_client):
        import asyncio

        agent = TesterAgent(client=mock_llm_client)

        mock_proc = MagicMock()
        mock_proc.communicate = AsyncMock(return_value=(b"5 passed", b""))

        with patch(
            "furrow.agents.tester.asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ):
            with patch(
                "furrow.agents.tester.asyncio.wait_for",
                return_value=(b"5 passed", b""),
            ):
                result = await agent._run_tests()

        assert "5 passed" in result
