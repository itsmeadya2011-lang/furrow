"""Tests for the WorkerAgent."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


@pytest.fixture
def mock_client(tmp_path):
    client = MagicMock()
    client.settings.workspace = tmp_path
    client.complete = AsyncMock(return_value="Implemented the task successfully.")

    async def _read_file(path):
        return Path(path).read_text()

    client.read_file = AsyncMock(side_effect=_read_file)
    return client


class TestWorkerRun:
    @pytest.mark.asyncio
    async def test_returns_llm_response(self, mock_client):
        task = TaskModel(id="1", description="Create a new module", files=["module.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result == "Implemented the task successfully."

    @pytest.mark.asyncio
    async def test_passes_task_description_in_prompt(self, mock_client):
        task = TaskModel(id="1", description="Add JWT auth", files=["auth.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        args, kwargs = mock_client.complete.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert "Add JWT auth" in prompt
        assert "auth.py" in prompt

    @pytest.mark.asyncio
    async def test_no_files_specified(self, mock_client):
        task = TaskModel(id="1", description="General task", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        assert result is not None

        args, kwargs = mock_client.complete.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert "any" in prompt

    @pytest.mark.asyncio
    async def test_reads_context_files(self, mock_client, tmp_path):
        """WorkerAgent should read existing files listed in the task to provide context."""
        existing_file = tmp_path / "existing.py"
        existing_file.write_text("def old_function(): pass")

        task = TaskModel(id="1", description="Update function", files=["existing.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        # Should have attempted to read the file
        mock_client.read_file.assert_called_once()
        read_path = mock_client.read_file.call_args[0][0]
        assert "existing.py" in str(read_path)

    @pytest.mark.asyncio
    async def test_nonexistent_file_in_context(self, mock_client, tmp_path):
        """If a file doesn't exist, context should note it rather than crashing."""
        task = TaskModel(id="1", description="Create new file", files=["new_file.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()
        # Should not crash, read_file should be called
        assert result is not None

    @pytest.mark.asyncio
    async def test_prompt_includes_file_context(self, mock_client, tmp_path):
        """The file context should be included in the prompt sent to the LLM."""
        existing_file = tmp_path / "config.py"
        existing_file.write_text("DEBUG = True")

        task = TaskModel(id="1", description="Update config", files=["config.py"])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        args, kwargs = mock_client.complete.call_args
        prompt = args[0] if args else kwargs.get("prompt", "")
        assert "=== Existing file context ===" in prompt
        assert "DEBUG = True" in prompt
