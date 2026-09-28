"""Tests for WorkerAgent file context building and task execution."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.agents.prompts import WORKER_PROMPT
from furrow.config import TaskModel


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "sonnet"
    client.read_file = AsyncMock()
    client.complete = AsyncMock(return_value='{"status": "completed", "summary": "done", "changes": []}')
    return client


@pytest.fixture
def temp_file(tmp_path):
    """Create a temporary file and return its path."""
    f = tmp_path / "test_file.py"
    f.write_text("def hello():\n    pass\n")
    return str(f)


class TestBuildFilesContext:
    @pytest.mark.asyncio
    async def test_reads_existing_file(self, mock_client, temp_file):
        mock_client.read_file = AsyncMock(return_value="file content here")
        task = TaskModel(id="1", description="test", files=[temp_file])
        worker = WorkerAgent(task=task, client=mock_client)
        context = await worker._build_files_context()
        assert temp_file in context
        assert "file content here" in context
        assert "--- File:" in context

    @pytest.mark.asyncio
    async def test_handles_nonexistent_file(self, mock_client):
        mock_client.read_file = AsyncMock(side_effect=FileNotFoundError("not found"))
        task = TaskModel(id="1", description="test", files=["nonexistent.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        context = await worker._build_files_context()
        assert "nonexistent.py" in context
        assert "[File does not exist]" in context

    @pytest.mark.asyncio
    async def test_no_files(self, mock_client):
        task = TaskModel(id="1", description="test", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        context = await worker._build_files_context()
        assert context == ""


class TestWorkerRun:
    @pytest.mark.asyncio
    async def test_run_includes_file_context(self, mock_client, temp_file):
        mock_client.read_file = AsyncMock(return_value="content")
        task = TaskModel(id="1", description="Add a function", files=[temp_file])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        mock_client.complete.assert_called_once()
        prompt = mock_client.complete.call_args[0][0]
        assert "Add a function" in prompt
        assert "File contents provided for context" in prompt
        assert "content" in prompt

    @pytest.mark.asyncio
    async def test_run_calls_complete_with_worker_model(self, mock_client):
        task = TaskModel(id="1", description="test", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()
        mock_client.complete.assert_called_once()
        kwargs = mock_client.complete.call_args
        assert kwargs[1]["model"] == "sonnet"
