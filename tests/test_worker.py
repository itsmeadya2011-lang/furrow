import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from furrow.agents.worker import WorkerAgent, _build_tools
from furrow.config import TaskModel


def test_worker_init():
    task = TaskModel(id="1", description="test task")
    mock_client = Mock()
    worker = WorkerAgent(task=task, client=mock_client)
    assert worker.task == task
    assert worker.client == mock_client
    assert worker.max_iterations == 10


def test_build_tools():
    tools = _build_tools()
    assert len(tools) == 4
    tool_names = [t["name"] for t in tools]
    assert "read_file" in tool_names
    assert "write_file" in tool_names
    assert "list_files" in tool_names
    assert "run_command" in tool_names


@pytest.mark.asyncio
async def test_execute_tool_read_file(tmp_path):
    task = TaskModel(id="1", description="test")
    mock_client = Mock()
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello World")
    mock_client.read_file = AsyncMock(return_value="Hello World")
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker._execute_tool("read_file", {"path": str(test_file)})
    assert result == "Hello World"
    mock_client.read_file.assert_called_once()


@pytest.mark.asyncio
async def test_execute_tool_write_file(tmp_path):
    task = TaskModel(id="1", description="test")
    mock_client = Mock()
    mock_client.write_file = AsyncMock()
    worker = WorkerAgent(task=task, client=mock_client)
    test_file = tmp_path / "output.txt"
    result = await worker._execute_tool("write_file", {"path": str(test_file), "content": "test content"})
    assert "Wrote 12 bytes" in result
    mock_client.write_file.assert_called_once()


@pytest.mark.asyncio
async def test_execute_tool_list_files(tmp_path):
    task = TaskModel(id="1", description="test")
    mock_client = Mock()
    mock_client.list_files = Mock(return_value=["file1.txt", "subdir/file2.py"])
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker._execute_tool("list_files", {"directory": str(tmp_path)})
    assert "file1.txt" in result
    assert "subdir/file2.py" in result


@pytest.mark.asyncio
async def test_execute_tool_run_command():
    task = TaskModel(id="1", description="test")
    mock_client = Mock()
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker._execute_tool("run_command", {"command": "echo hello"})
    assert "hello" in result


@pytest.mark.asyncio
async def test_execute_tool_unknown():
    task = TaskModel(id="1", description="test")
    mock_client = Mock()
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker._execute_tool("unknown_tool", {})
    assert "Unknown tool" in result


@pytest.mark.asyncio
async def test_execute_tool_error():
    task = TaskModel(id="1", description="test")
    mock_client = Mock()
    mock_client.read_file = AsyncMock(side_effect=Exception("File not found"))
    worker = WorkerAgent(task=task, client=mock_client)
    result = await worker._execute_tool("read_file", {"path": "/nonexistent/file.txt"})
    assert "Error executing read_file" in result
    assert "File not found" in result
