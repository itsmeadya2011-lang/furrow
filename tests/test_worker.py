import pytest
from unittest.mock import MagicMock, AsyncMock

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


def test_extract_tool_actions_single():
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = 'Some text\n{"action": "read_file", "path": "/tmp/test.py"}\nMore text'
    actions = worker._extract_tool_actions(response)
    assert len(actions) == 1
    assert actions[0]["action"] == "read_file"


def test_extract_tool_actions_multiple():
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = '{"action": "read_file", "path": "/tmp/a.py"}\n{"action": "write_file", "path": "/tmp/b.py"}'
    actions = worker._extract_tool_actions(response)
    assert len(actions) == 2
    assert actions[0]["action"] == "read_file"
    assert actions[1]["action"] == "write_file"


def test_extract_tool_actions_none():
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = "Just some plain text without any JSON blocks"
    actions = worker._extract_tool_actions(response)
    assert actions == []


def test_extract_tool_actions_ignores_non_json():
    worker = WorkerAgent(task=TaskModel(id="1", description="test"))
    response = 'Some text\n{"not_action": "value"}\n{"action": "read_file"}\nNot JSON: {invalid}'
    actions = worker._extract_tool_actions(response)
    assert len(actions) == 1
    assert actions[0]["action"] == "read_file"


@pytest.mark.asyncio
async def test_execute_tool_read_file():
    client = MagicMock()
    client.read_file = AsyncMock(return_value="file content")
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
    result = await worker._execute_tool("read_file", {"path": "/tmp/test.txt"})
    assert result == "--- /tmp/test.txt ---\nfile content"
    client.read_file.assert_called_once_with("/tmp/test.txt")


@pytest.mark.asyncio
async def test_execute_tool_write_file():
    client = MagicMock()
    client.write_file = AsyncMock()
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
    result = await worker._execute_tool("write_file", {"path": "/tmp/test.txt", "content": "hello"})
    assert result == "Wrote 5 bytes to /tmp/test.txt"
    client.write_file.assert_called_once_with("/tmp/test.txt", "hello")


@pytest.mark.asyncio
async def test_execute_tool_list_files():
    client = MagicMock()
    client.list_files = MagicMock(return_value=["a.py", "b.py"])
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
    result = await worker._execute_tool("list_files", {"path": "/tmp"})
    assert result == "a.py\nb.py"
    client.list_files.assert_called_once_with("/tmp")


@pytest.mark.asyncio
async def test_execute_tool_run_shell():
    client = MagicMock()
    client.run_shell = AsyncMock(return_value=(0, "output", ""))
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
    result = await worker._execute_tool("run_shell", {"command": "echo hi"})
    assert "exit code: 0" in result
    assert "stdout:\noutput" in result
    client.run_shell.assert_called_once_with("echo hi")


@pytest.mark.asyncio
async def test_execute_tool_unknown():
    client = MagicMock()
    worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
    result = await worker._execute_tool("unknown_action", {})
    assert result == "Unknown action: unknown_action"
