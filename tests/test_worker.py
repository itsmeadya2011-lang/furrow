from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


@pytest.mark.asyncio
async def test_worker_parses_json_response():
    """Worker should parse JSON response with file changes."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "files": [
            {"path": "src/example.py", "content": "def hello():\n    return 'world'\n"}
        ],
        "summary": "Created src/example.py"
    }))
    mock_client.read_file = AsyncMock(side_effect=FileNotFoundError())
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Create example module", files=["src/example.py"])
    agent = WorkerAgent(task=task, client=mock_client)

    result = await agent.run()

    assert "src/example.py" in result
    mock_client.write_file.assert_called_once()
    args = mock_client.write_file.call_args
    assert args[0][0] == "src/example.py"


@pytest.mark.asyncio
async def test_worker_parses_fenced_json():
    """Worker should extract JSON from code fences."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value='```json\n{"files": [{"path": "test.py", "content": "x = 1"}], "summary": "ok"}\n```')
    mock_client.read_file = AsyncMock(side_effect=FileNotFoundError())
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Create test file", files=["test.py"])
    agent = WorkerAgent(task=task, client=mock_client)

    result = await agent.run()

    assert "test.py" in result
    mock_client.write_file.assert_called_once()


@pytest.mark.asyncio
async def test_worker_no_files_in_response():
    """Worker should handle response with no parseable file changes."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value="I couldn't generate any changes.")
    mock_client.read_file = AsyncMock(side_effect=FileNotFoundError())
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Do something", files=["a.py"])
    agent = WorkerAgent(task=task, client=mock_client)

    result = await agent.run()

    assert "No file changes" in result
    mock_client.write_file.assert_not_called()


@pytest.mark.asyncio
async def test_worker_reads_existing_files():
    """Worker should read files before asking LLM."""
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=json.dumps({
        "files": [{"path": "existing.py", "content": "# updated"}],
        "summary": "updated"
    }))
    mock_client.read_file = AsyncMock(return_value="# original content")
    mock_client.write_file = AsyncMock()

    task = TaskModel(id="1", description="Update file", files=["existing.py"])
    agent = WorkerAgent(task=task, client=mock_client)

    await agent.run()

    mock_client.read_file.assert_called_once_with("existing.py")
    # The prompt should include the file content
    call_args = mock_client.complete.call_args
    assert "original content" in call_args[0][0]