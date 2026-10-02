"""Tests for the WorkerAgent."""

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class FakeLLMClient:
    """Fake LLM client that returns scripted responses."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.call_count = 0
        self.settings = type(
            "S",
            (),
            {
                "workspace": ".",
                "worker_model": "fake",
                "max_worker_iterations": 10,
            },
        )()

    async def converse(self, messages: list[dict[str, str]]) -> str:
        if self.call_count < len(self.responses):
            resp = self.responses[self.call_count]
            self.call_count += 1
            return resp
        return '{"tool": "done", "summary": "No more responses"}'

    async def read_file_safe(self, path):
        return None

    async def write_file(self, path, content):
        pass

    async def list_files(self, directory):
        return []


@pytest.fixture
def simple_task():
    return TaskModel(id="1", description="Add a hello function to main.py")


def test_worker_init(simple_task):
    """WorkerAgent stores task and initializes client."""
    worker = WorkerAgent(task=simple_task)
    assert worker.task == simple_task
    assert worker.task.id == "1"


def test_extract_tool_call_no_json():
    """_extract_tool_calls returns empty list for plain text."""
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    result = worker._extract_tool_calls("This is just plain text with no tools.")
    assert result == []


def test_extract_tool_call_single_json():
    """_extract_tool_calls extracts a single JSON tool call."""
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = '{"tool": "done", "summary": "Task complete"}'
    result = worker._extract_tool_calls(response)
    assert len(result) == 1
    assert result[0]["tool"] == "done"
    assert result[0]["summary"] == "Task complete"


def test_extract_tool_call_in_code_block():
    """_extract_tool_calls extracts JSON from code blocks."""
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = '```tool\n{"tool": "read_file", "path": "main.py"}\n```'
    result = worker._extract_tool_calls(response)
    assert len(result) == 1
    assert result[0]["tool"] == "read_file"


def test_extract_tool_call_in_json_block():
    """_extract_tool_calls extracts JSON from ```json blocks."""
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = '```json\n{"tool": "write_file", "path": "x.py", "content": "hello"}\n```'
    result = worker._extract_tool_calls(response)
    assert len(result) == 1
    assert result[0]["tool"] == "write_file"


def test_extract_tool_call_multiple():
    """_extract_tool_calls extracts multiple tool calls."""
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = '{"tool": "read_file", "path": "a.py"}\n{"tool": "read_file", "path": "b.py"}'
    result = worker._extract_tool_calls(response)
    assert len(result) == 2


def test_extract_tool_call_ignores_non_tool_json():
    """_extract_tool_calls ignores JSON without a 'tool' key."""
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = '{"not_tool": "value", "other": 123}'
    result = worker._extract_tool_calls(response)
    assert result == []


@pytest.mark.asyncio
async def test_worker_done_immediately(simple_task):
    """Worker returns summary when LLM immediately calls done."""
    client = FakeLLMClient(['{"tool": "done", "summary": "Added hello function"}'])
    worker = WorkerAgent(task=simple_task, client=client)
    result = await worker.run()
    assert result == "Added hello function"


@pytest.mark.asyncio
async def test_worker_no_tools_treats_as_done(simple_task):
    """Worker treats plain text response (no tool calls) as final answer."""
    client = FakeLLMClient(["This is my final summary"])
    worker = WorkerAgent(task=simple_task, client=client)
    result = await worker.run()
    assert "This is my final summary" in result


@pytest.mark.asyncio
async def test_worker_respects_max_iterations(simple_task):
    """Worker stops after max_iterations even if not done."""
    client = FakeLLMClient(['{"tool": "list_files", "path": "."}'] * 20)
    client.settings.max_worker_iterations = 3
    worker = WorkerAgent(task=simple_task, client=client, max_iterations=3)
    result = await worker.run()
    assert isinstance(result, str)
