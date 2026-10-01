"""Tests for the WorkerAgent — file parsing and writing."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


@pytest.fixture
def client():
    c = MagicMock()
    c.settings.worker_model = "model"
    c.complete = AsyncMock()
    c.read_file = AsyncMock()
    c.write_file = AsyncMock()
    return c


@pytest.fixture
def task():
    return TaskModel(id="1", description="Create auth module", files=["src/auth.py"])


@pytest.mark.asyncio
async def test_worker_writes_files_from_json(client, task):
    """Worker should parse JSON response and write files to disk."""
    response = json.dumps({
        "files": [
            {"path": "src/auth.py", "content": "def login(): pass"},
        ],
        "summary": "Created auth module",
    })
    client.complete.return_value = response

    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "Created auth module"
    client.write_file.assert_awaited_once_with("src/auth.py", "def login(): pass")


@pytest.mark.asyncio
async def test_worker_handles_markdown_fenced_json(client, task):
    """Worker should strip markdown fences before parsing JSON."""
    response = '```json\n' + json.dumps({
        "files": [
            {"path": "src/auth.py", "content": "code"},
        ],
        "summary": "done",
    }) + '\n```'
    client.complete.return_value = response

    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "done"
    client.write_file.assert_awaited_once_with("src/auth.py", "code")


@pytest.mark.asyncio
async def test_worker_writes_multiple_files(client, task):
    response = json.dumps({
        "files": [
            {"path": "src/auth.py", "content": "code1"},
            {"path": "tests/test_auth.py", "content": "code2"},
        ],
        "summary": "Created both files",
    })
    client.complete.return_value = response

    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert "Created both files" in result
    assert client.write_file.await_count == 2
    client.write_file.assert_any_await("src/auth.py", "code1")
    client.write_file.assert_any_await("tests/test_auth.py", "code2")


@pytest.mark.asyncio
async def test_worker_reads_existing_files_for_context(client, task):
    """Worker should read existing task files to include in the prompt."""
    client.read_file.return_value = "existing content"
    client.complete.return_value = json.dumps({
        "files": [],
        "summary": "Updated",
    })

    worker = WorkerAgent(task=task, client=client)
    await worker.run()

    client.read_file.assert_awaited_with("src/auth.py")


@pytest.mark.asyncio
async def test_worker_handles_empty_response(client, task):
    """Worker should fall back gracefully when LLM returns non-JSON."""
    client.complete.return_value = "Just some prose, no JSON."

    worker = WorkerAgent(task=task, client=client)
    await worker.run()

    client.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_no_files_to_touch(client):
    """Worker should handle tasks with no specific files."""
    task = TaskModel(id="2", description="General cleanup")
    client.complete.return_value = json.dumps({
        "files": [{"path": "README.md", "content": "updated"}],
        "summary": "Updated README",
    })

    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "Updated README"
    client.write_file.assert_awaited_once_with("README.md", "updated")


@pytest.mark.asyncio
async def test_worker_skips_files_without_path(client, task):
    """Worker should skip file entries that have no path."""
    client.complete.return_value = json.dumps({
        "files": [
            {"path": "src/auth.py", "content": "code"},
            {"content": "missing path"},
        ],
        "summary": "done",
    })

    worker = WorkerAgent(task=task, client=client)
    await worker.run()

    # Should only write the file with a valid path.
    assert client.write_file.await_count == 1
    client.write_file.assert_awaited_once_with("src/auth.py", "code")
