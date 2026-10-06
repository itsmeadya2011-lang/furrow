import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


@pytest.mark.asyncio
async def test_worker_returns_done_summary():
    client = MagicMock()
    client.settings.worker_model = "test"
    client.complete = AsyncMock(return_value='{"tool": "done", "summary": "finished"}')
    worker = WorkerAgent(task=TaskModel(id="1", description="do work"), client=client)
    result = await worker.run()
    assert result == "finished"


@pytest.mark.asyncio
async def test_worker_returns_plain_text_when_no_json():
    client = MagicMock()
    client.settings.worker_model = "test"
    client.complete = AsyncMock(return_value="I implemented the feature.")
    worker = WorkerAgent(task=TaskModel(id="1", description="do work"), client=client)
    result = await worker.run()
    assert result == "I implemented the feature."


@pytest.mark.asyncio
async def test_worker_parses_tool_call_with_fences():
    client = MagicMock()
    client.settings.worker_model = "test"
    client.complete = AsyncMock(return_value='```json\n{"tool": "done", "summary": "finished"}\n```')
    worker = WorkerAgent(task=TaskModel(id="1", description="do work"), client=client)
    result = await worker.run()
    assert result == "finished"


@pytest.mark.asyncio
async def test_worker_executes_read_write_rounds():
    client = MagicMock()
    client.settings.worker_model = "test"
    client.complete = AsyncMock(side_effect=[
        '{"tool": "read_file", "path": "test.txt"}',
        '{"tool": "write_file", "path": "test.txt", "content": "hello"}',
        '{"tool": "done", "summary": "done"}',
    ])
    client.list_files = MagicMock(return_value=[])
    client.read_file = AsyncMock(return_value="old content")
    client.write_file = AsyncMock()

    worker = WorkerAgent(task=TaskModel(id="1", description="do work"), client=client)
    result = await worker.run()
    assert result == "done"
    client.write_file.assert_called_once_with("test.txt", "hello")
    client.read_file.assert_called_once_with("test.txt")


@pytest.mark.asyncio
async def test_worker_executes_list_files():
    client = MagicMock()
    client.settings.worker_model = "test"
    client.complete = AsyncMock(side_effect=[
        '{"tool": "list_files", "path": "."}',
        '{"tool": "done", "summary": "listed"}',
    ])
    client.list_files = MagicMock(return_value=["a.txt", "b.py"])
    client.read_file = AsyncMock()
    client.write_file = AsyncMock()

    worker = WorkerAgent(task=TaskModel(id="1", description="do work"), client=client)
    result = await worker.run()
    assert result == "listed"
    client.list_files.assert_called_once_with(".")


@pytest.mark.asyncio
async def test_worker_max_rounds():
    client = MagicMock()
    client.settings.worker_model = "test"
    client.complete = AsyncMock(return_value='{"tool": "read_file", "path": "x"}')
    client.list_files = MagicMock(return_value=[])

    worker = WorkerAgent(task=TaskModel(id="1", description="do work"), client=client)
    result = await worker.run()
    assert "max rounds" in result
