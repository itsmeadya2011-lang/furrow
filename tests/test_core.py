import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.llm import LLMClient
from furrow.agents.worker import WorkerAgent, FILE_TOOLS
from furrow.core.orchestrator import Orchestrator
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_anthropic_tool_use_loop():
    with patch('furrow.llm.AsyncAnthropic') as MockAsyncAnthropic:
        mock_anthropic_instance = AsyncMock()
        MockAsyncAnthropic.return_value = mock_anthropic_instance
        
        # Prepare responses
        # First response: tool use
        block1 = MagicMock()
        block1.type = "tool_use"
        block1.id = "tool1"
        block1.name = "read_file"
        block1.input = {"path": "test.txt"}
        mock_response1 = MagicMock()
        mock_response1.stop_reason = "tool_use"
        mock_response1.content = [block1]
        
        # Second response: final text
        text_block = MagicMock()
        text_block.type = "text"
        text_block.text = "Final answer"
        mock_response2 = MagicMock()
        mock_response2.stop_reason = "end_turn"
        mock_response2.content = [text_block]
        
        responses = [mock_response1, mock_response2]
        
        async def mock_create(*args, **kwargs):
            return responses.pop(0)
        
        mock_anthropic_instance.messages.create.side_effect = mock_create

        client = LLMClient()
        client.settings.provider = Provider.ANTHROPIC
        client.settings.anthropic_api_key = "test-key"

        # Define a tool handler as a mock
        read_file_handler = AsyncMock(return_value="file contents")

        tool_handlers = {"read_file": read_file_handler}
        tools = [
            {
                "name": "read_file",
                "description": "Read a file",
                "input_schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            }
        ]

        result = await client.complete_with_tools(
            prompt="Read the file",
            tools=tools,
            tool_handlers=tool_handlers,
        )

        # Verify the tool handler was called with the correct args
        read_file_handler.assert_called_once_with({"path": "test.txt"})
        # Verify the final result is the text from the second response
        assert result == "Final answer"
        # Verify that messages.create was called twice
        assert mock_anthropic_instance.messages.create.call_count == 2


@pytest.mark.asyncio
async def test_openai_tool_use_loop():
    with patch('furrow.llm.AsyncOpenAI') as MockAsyncOpenAI:
        mock_openai_instance = AsyncMock()
        MockAsyncOpenAI.return_value = mock_openai_instance
        
        # First response: tool call
        tool_call = MagicMock()
        tool_call.id = "call1"
        tool_call.function = MagicMock()
        tool_call.function.name = "write_file"
        tool_call.function.arguments = '{"path": "out.txt", "content": "hello"}'
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.tool_calls = [tool_call]
        mock_response1.choices[0].message.content = None
        
        # Second response: final text
        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.tool_calls = None
        mock_response2.choices[0].message.content = "Final answer"
        
        responses = [mock_response1, mock_response2]
        
        async def mock_create(*args, **kwargs):
            return responses.pop(0)
        
        mock_openai_instance.chat.completions.create.side_effect = mock_create

        client = LLMClient()
        client.settings.provider = Provider.OPENAI
        client.settings.openai_api_key = "test-key"

        # Define a tool handler as a mock
        write_file_handler = AsyncMock(return_value="written")

        tool_handlers = {"write_file": write_file_handler}
        tools = [
            {
                "name": "write_file",
                "description": "Write a file",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
            }
        ]

        result = await client.complete_with_tools(
            prompt="Write a file",
            tools=tools,
            tool_handlers=tool_handlers,
        )

        write_file_handler.assert_called_once_with({"path": "out.txt", "content": "hello"})
        assert result == "Final answer"
        assert mock_openai_instance.chat.completions.create.call_count == 2


def test_worker_agent_run_wiring():
    # Create a mock LLMClient that records the arguments to complete_with_tools
    mock_client = MagicMock()
    mock_client.complete_with_tools = AsyncMock(return_value="done")

    # We need to mock the client's settings.worker_model
    mock_client.settings = MagicMock()
    mock_client.settings.worker_model = "test-model"

    # Create a WorkerAgent with a dummy task
    task = TaskModel(id="1", description="test task", files=[])
    worker = WorkerAgent(task=task, client=mock_client)

    # Run the worker (we'll run the async method in a sync test by using asyncio.run)
    import asyncio
    result = asyncio.run(worker.run())

    # Verify that complete_with_tools was called with the expected tools and tool_handlers
    mock_client.complete_with_tools.assert_called_once()
    call_args, call_kwargs = mock_client.complete_with_tools.call_args
    assert call_kwargs["tools"] == FILE_TOOLS
    assert set(call_kwargs["tool_handlers"].keys()) == {"read_file", "write_file", "list_files"}
    assert result == "done"


@pytest.mark.asyncio
async def test_orchestrator_loop_fix():
    # Create a fake planner that returns a plan with 1 task
    class FakePlanner:
        def __init__(self, client=None):
            pass
        async def plan(self, goal):
            from furrow.config import Plan, TaskModel
            return Plan(
                tasks=[TaskModel(id="1", description="test task", files=[])],
                rationale="test"
            )

    # Create a fake worker that returns "ok"
    class FakeWorker:
        def __init__(self, task, client=None):
            self.task = task
        async def run(self):
            return "ok"

    # Create a fake tester that returns a passed TestResult
    class FakeTester:
        def __init__(self, client=None):
            pass
        async def run(self, goal, tasks):
            from furrow.config import TestResult
            return TestResult(passed=True, summary="all passed", failures=[])

    # Patch the Orchestrator's internal agents to use our fakes
    with patch('furrow.core.orchestrator.PlannerAgent', FakePlanner), \
         patch('furrow.core.orchestrator.WorkerAgent', FakeWorker), \
         patch('furrow.core.orchestrator.TesterAgent', FakeTester):
        
        orchestrator = Orchestrator(goal="test goal")
        # Run one cycle
        await orchestrator._cycle()

        # After the cycle, _is_done should return True because the task is completed and tests passed
        assert orchestrator._is_done() == True
        # _get_tasks should return the persisted tasks (not empty)
        tasks = orchestrator._get_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == "1"
        assert tasks[0].status == "completed"

    # Test that max_cycles halts the loop
    with patch('furrow.core.orchestrator.PlannerAgent', FakePlanner), \
         patch('furrow.core.orchestrator.WorkerAgent', FakeWorker), \
         patch('furrow.core.orchestrator.TesterAgent', FakeTester):
        
        orchestrator = Orchestrator(goal="test goal")
        orchestrator.settings.max_cycles = 1
        # Run one cycle (which will increment cycles to 1)
        await orchestrator._cycle()
        # Now _is_done should return True because max_cycles reached
        assert orchestrator._is_done() == True