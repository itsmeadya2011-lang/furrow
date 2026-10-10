from unittest.mock import AsyncMock, Mock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


def test_worker_init():
    task = TaskModel(id="1", description="test task")
    worker = WorkerAgent(task=task)
    assert worker.task == task
    assert worker.max_iterations == 10
    assert isinstance(worker.client, LLMClient)


def test_worker_init_with_client():
    task = TaskModel(id="1", description="test task")
    mock_client = Mock(spec=LLMClient)
    worker = WorkerAgent(task=task, client=mock_client)
    assert worker.client == mock_client


@pytest.mark.asyncio
async def test_worker_run_simple_completion():
    """Worker returns text when LLM responds without tool calls (Anthropic)."""
    settings = Settings(_env_file=None)
    client = Mock(spec=LLMClient)
    client.settings = settings

    mock_response = Mock()
    mock_response.content = [Mock(type="text", text="Task done!")]
    client.complete_with_tools = AsyncMock(return_value=mock_response)

    task = TaskModel(id="1", description="Do something")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Task done!"


@pytest.mark.asyncio
async def test_worker_run_openai_completion():
    """Worker returns text when LLM responds without tool calls (OpenAI)."""
    settings = Settings(_env_file=None)
    client = Mock(spec=LLMClient)
    client.settings = settings

    mock_response = Mock()
    mock_message = Mock()
    mock_message.content = "OpenAI task done"
    mock_message.tool_calls = None
    mock_choice = Mock()
    mock_choice.message = mock_message
    mock_response.choices = [mock_choice]
    client.complete_with_tools = AsyncMock(return_value=mock_response)

    task = TaskModel(id="1", description="Do something")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "OpenAI task done"


@pytest.mark.asyncio
async def test_worker_run_with_tool_use():
    """Worker executes tools and continues until no more tool calls."""
    settings = Settings(_env_file=None)
    client = Mock(spec=LLMClient)
    client.settings = settings

    # First response: tool use (read_file)
    tool_use_block = Mock()
    tool_use_block.configure_mock(type="tool_use", id="tu_1", name="read_file", input={"path": "test.txt"})
    first_response = Mock()
    text_block = Mock()
    text_block.configure_mock(type="text", text="Reading file")
    first_response.content = [text_block, tool_use_block]

    # Second response: final text
    second_response = Mock()
    final_text_block = Mock()
    final_text_block.configure_mock(type="text", text="All done")
    second_response.content = [final_text_block]

    client.complete_with_tools = AsyncMock(side_effect=[first_response, second_response])
    client.read_file = AsyncMock(return_value="file contents")

    task = TaskModel(id="1", description="Read and report")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "All done"
    client.read_file.assert_called_once_with("test.txt")


@pytest.mark.asyncio
async def test_worker_run_unknown_tool():
    """Worker handles unknown tools gracefully."""
    settings = Settings(_env_file=None)
    client = Mock(spec=LLMClient)
    client.settings = settings

    tool_use_block = Mock(type="tool_use", id="tu_1", name="unknown_tool", input={})
    first_response = Mock()
    first_response.content = [tool_use_block]

    second_response = Mock()
    second_response.content = [Mock(type="text", text="Done after unknown tool")]

    client.complete_with_tools = AsyncMock(side_effect=[first_response, second_response])

    task = TaskModel(id="1", description="Test unknown tool")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Done after unknown tool"


@pytest.mark.asyncio
async def test_worker_run_max_iterations():
    """Worker stops after max_iterations even if tools keep being called."""
    settings = Settings(_env_file=None)
    client = Mock(spec=LLMClient)
    client.settings = settings

    tool_use_block = Mock(type="tool_use", id="tu_1", name="read_file", input={"path": "x"})
    loop_response = Mock()
    loop_response.content = [tool_use_block]

    client.complete_with_tools = AsyncMock(return_value=loop_response)
    client.read_file = AsyncMock(return_value="data")

    task = TaskModel(id="1", description="Loop task")
    worker = WorkerAgent(task=task, client=client)
    worker.max_iterations = 3
    result = await worker.run()
    # After max iterations, returns the last text or default
    assert result == "Task completed."
    assert client.complete_with_tools.call_count == 3