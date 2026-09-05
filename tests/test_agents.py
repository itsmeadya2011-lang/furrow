"""Tests for the Agent base class and agent subclasses."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.agents.base import Agent
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient, LLMResult, ToolCall
from furrow.tools.base import ToolResult


class TestAgentBase:
    def test_base_has_no_tools_by_default(self):
        assert Agent.tools == []

    def test_base_system_prompt_default(self):
        assert Agent.system_prompt == ""

    def test_agent_init_default_client(self):
        with patch("furrow.agents.base.LLMClient") as MockClient:
            MockClient.return_value = MagicMock()
            agent = Agent()
            assert agent.client is not None


class TestAgentRunLoop:
    @pytest.mark.asyncio
    async def test_run_loop_no_tool_calls(self):
        """When the LLM returns no tool calls, the loop ends immediately."""
        agent = Agent()
        agent.system_prompt = "You are a test."
        result = LLMResult(text="Done!", tool_calls=[])
        with patch.object(agent.client, "complete_with_tools", new_callable=AsyncMock, return_value=result):
            res = await agent.run_loop("test prompt", max_iterations=5)

        assert res.text == "Done!"
        assert res.tool_calls == []

    @pytest.mark.asyncio
    async def test_run_loop_with_tool_calls(self):
        """Agent calls a tool, gets result, then finishes."""
        from furrow.tools.file_ops import ReadFile

        agent = Agent()
        agent.system_prompt = "You are a test."
        agent.tools = [ReadFile]

        tool_call_result = LLMResult(
            text="Reading the file...",
            tool_calls=[ToolCall(id="call_1", name="read_file", input={"path": "/tmp/test.txt"})],
        )
        final_result = LLMResult(text="I've read the file. It contains important info.", tool_calls=[])

        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_call_result
            return final_result

        with patch.object(agent.client, "complete_with_tools", side_effect=mock_complete):
            with patch.object(
                ReadFile, "execute",
                new_callable=AsyncMock,
                return_value=ToolResult(success=True, output="file contents"),
            ):
                res = await agent.run_loop("Read /tmp/test.txt", max_iterations=5)

        assert call_count == 2
        assert "important info" in res.text
        assert res.tool_calls == []

    @pytest.mark.asyncio
    async def test_run_loop_max_iterations(self):
        """Agent stops after max_iterations even if tools keep being called."""
        from furrow.tools.execute import ExecuteCommand

        agent = Agent()
        agent.system_prompt = "You are a test."
        agent.tools = [ExecuteCommand]

        always_tool_call = LLMResult(
            text="Running...",
            tool_calls=[ToolCall(id="call_1", name="execute_command", input={"command": "echo hi"})],
        )

        mock_fn = AsyncMock(return_value=always_tool_call)
        with patch.object(agent.client, "complete_with_tools", mock_fn):
            with patch.object(
                ExecuteCommand, "execute",
                new_callable=AsyncMock,
                return_value=ToolResult(success=True, output="hi"),
            ):
                res = await agent.run_loop("do something", max_iterations=3)

        assert res.text == "Running..."
        assert mock_fn.call_count == 3

    @pytest.mark.asyncio
    async def test_run_loop_unknown_tool(self):
        """Agent handles unknown tool calls gracefully."""
        agent = Agent()
        agent.system_prompt = "Test."
        agent.tools = []

        with_tool_call = LLMResult(
            text="Using a tool.",
            tool_calls=[ToolCall(id="call_1", name="unknown_tool", input={})],
        )
        no_more = LLMResult(text="Done.", tool_calls=[])

        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return with_tool_call
            return no_more

        with patch.object(agent.client, "complete_with_tools", side_effect=mock_complete):
            res = await agent.run_loop("test", max_iterations=5)

        assert call_count == 2
        assert res.text == "Done."

    @pytest.mark.asyncio
    async def test_run_loop_multiple_concurrent_tools(self):
        """Agent handles multiple tool calls in one turn."""
        from furrow.tools.file_ops import ReadFile, WriteFile

        agent = Agent()
        agent.system_prompt = "Test."
        agent.tools = [ReadFile, WriteFile]

        tool_call_result = LLMResult(
            text="Calling tools.",
            tool_calls=[
                ToolCall(id="call_1", name="read_file", input={"path": "/tmp/a.txt"}),
                ToolCall(id="call_2", name="write_file", input={"path": "/tmp/b.txt", "content": "hi"}),
            ],
        )
        final_result = LLMResult(text="Done.", tool_calls=[])

        call_count = 0

        async def mock_complete(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return tool_call_result
            return final_result

        with patch.object(agent.client, "complete_with_tools", side_effect=mock_complete):
            with patch.object(
                ReadFile, "execute",
                new_callable=AsyncMock,
                return_value=ToolResult(success=True, output="content_a"),
            ):
                with patch.object(
                    WriteFile, "execute",
                    new_callable=AsyncMock,
                    return_value=ToolResult(success=True, output="written"),
                ):
                    res = await agent.run_loop("test", max_iterations=5)

        assert call_count == 2
        assert res.text == "Done."


class TestWorkerAgent:
    def test_worker_has_tools(self):
        assert len(WorkerAgent.tools) > 0
        tool_names = [t.name for t in WorkerAgent.tools]
        assert "read_file" in tool_names
        assert "write_file" in tool_names
        assert "execute_command" in tool_names

    def test_worker_init_with_task(self):
        task = TaskModel(id="1", description="Write a function", files=["src/test.py"])
        agent = WorkerAgent(task=task)
        assert agent.task is task
        from furrow.agents.prompts import WORKER_PROMPT
        assert agent.system_prompt == WORKER_PROMPT

    @pytest.mark.asyncio
    async def test_worker_run(self):
        task = TaskModel(id="1", description="Add a test", files=["test.py"])
        agent = WorkerAgent(task=task)

        result = LLMResult(text="I added comprehensive tests for the user model.", tool_calls=[])

        with patch.object(
            agent.client, "complete_with_tools",
            new_callable=AsyncMock, return_value=result,
        ):
            summary = await agent.run()

        assert "comprehensive tests" in summary


class TestPlannerAgent:
    def test_planner_has_no_tools(self):
        assert PlannerAgent.tools == []

    @pytest.mark.asyncio
    async def test_planner_parses_valid_json(self):
        agent = PlannerAgent()
        response = '{"tasks": [{"id": "1", "description": "do x", "files": [], "dependencies": []}], "rationale": "test"}'

        result = LLMResult(text=response, tool_calls=[])
        with patch.object(agent.client, "complete_with_tools", new_callable=AsyncMock, return_value=result):
            plan = await agent.plan("do something")

        from furrow.config import Plan
        assert isinstance(plan, Plan)
        assert len(plan.tasks) == 1
        assert plan.tasks[0].description == "do x"
        assert plan.rationale == "test"

    @pytest.mark.asyncio
    async def test_planner_strips_code_fences(self):
        agent = PlannerAgent()
        response = (
            '```json\n'
            '{"tasks": [{"id": "1", "description": "do x", "files": [], "dependencies": []}], "rationale": "test"}'
            '\n```'
        )

        result = LLMResult(text=response, tool_calls=[])
        with patch.object(agent.client, "complete_with_tools", new_callable=AsyncMock, return_value=result):
            plan = await agent.plan("do something")

        assert plan.tasks[0].description == "do x"

    @pytest.mark.asyncio
    async def test_planner_invalid_json_raises(self):
        agent = PlannerAgent()
        response = "not valid json"

        result = LLMResult(text=response, tool_calls=[])
        with patch.object(agent.client, "complete_with_tools", new_callable=AsyncMock, return_value=result):
            with pytest.raises(ValueError, match="Failed to parse plan"):
                await agent.plan("do something")


class TestTesterAgent:
    def test_tester_has_no_tools(self):
        assert TesterAgent.tools == []

    @pytest.mark.asyncio
    async def test_tester_parses_valid_json(self):
        agent = TesterAgent()
        response = '{"passed": true, "summary": "All tests passed", "failures": []}'

        result = LLMResult(text=response, tool_calls=[])
        with patch.object(agent.client, "complete_with_tools", new_callable=AsyncMock, return_value=result):
            test_result = await agent.run(
                "build feature", [TaskModel(id="1", description="t", status="completed")]
            )

        assert test_result.passed is True
        assert test_result.summary == "All tests passed"

    @pytest.mark.asyncio
    async def test_tester_handles_invalid_json(self):
        agent = TesterAgent()
        response = "Tests passed, looks good"

        result = LLMResult(text=response, tool_calls=[])
        with patch.object(agent.client, "complete_with_tools", new_callable=AsyncMock, return_value=result):
            test_result = await agent.run(
                "build feature", [TaskModel(id="1", description="t", status="completed")]
            )

        assert test_result.passed is True  # "passed" found in text

    @pytest.mark.asyncio
    async def test_tester_strips_code_fences(self):
        agent = TesterAgent()
        response = (
            '```json\n'
            '{"passed": false, "summary": "2 failures", "failures": ["error1", "error2"]}'
            '\n```'
        )

        result = LLMResult(text=response, tool_calls=[])
        with patch.object(agent.client, "complete_with_tools", new_callable=AsyncMock, return_value=result):
            test_result = await agent.run(
                "build feature", [TaskModel(id="1", description="t", status="completed")]
            )

        assert test_result.passed is False
        assert len(test_result.failures) == 2

    @pytest.mark.asyncio
    async def test_tester_runs_tests_and_summarizes(self):
        """Tester should call _run_tests and include output in the prompt."""
        agent = TesterAgent()
        summary_response = '{"passed": true, "summary": "All good", "failures": []}'
        result = LLMResult(text=summary_response, tool_calls=[])

        with patch.object(
            agent.client, "complete_with_tools",
            new_callable=AsyncMock, return_value=result,
        ) as mock:
            with patch.object(
                agent, "_run_tests",
                new_callable=AsyncMock, return_value="1 passed",
            ):
                test_result = await agent.run(
                    "test goal", [TaskModel(id="1", description="t", status="completed")]
                )

        assert test_result.passed is True
        # The test output should have been included in the prompt passed to complete_with_tools
        call_args = mock.call_args
        prompt = call_args.kwargs.get("prompt", "")
        assert "1 passed" in prompt
