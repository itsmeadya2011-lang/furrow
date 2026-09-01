from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_worker_parses_json_and_writes_files(self, tmp_path: Path) -> None:
        """When the LLM returns JSON with files, write_file is called for each and the summary is returned."""
        mock_client = AsyncMock()
        mock_client.settings.worker_model = "test-model"
        mock_client.settings.workspace = tmp_path
        mock_client.read_file = AsyncMock(return_value="")
        mock_client.list_files = MagicMock(return_value=[])
        mock_client.complete = AsyncMock(
            return_value=(
                '{"files": [{"path": "src/main.py", "content": "print(1)"}, '
                '{"path": "src/utils.py", "content": "def f(): pass"}], '
                '"summary": "Created two files"}'
            )
        )
        mock_client.write_file = AsyncMock()

        task = TaskModel(
            id="1",
            description="Create a new file",
            files=["src/main.py", "src/utils.py"],
        )
        agent = WorkerAgent(task=task, client=mock_client)

        result = await agent.run()

        assert mock_client.write_file.call_count == 2
        mock_client.write_file.assert_any_call("src/main.py", "print(1)")
        mock_client.write_file.assert_any_call("src/utils.py", "def f(): pass")
        assert "Created two files" in result

    @pytest.mark.asyncio
    async def test_worker_handles_parse_failure(self, tmp_path: Path) -> None:
        """When the LLM returns non-JSON, a fallback notes file is written and parse failure is reported."""
        mock_client = AsyncMock()
        mock_client.settings.worker_model = "test-model"
        mock_client.settings.workspace = tmp_path
        mock_client.read_file = AsyncMock(return_value="")
        mock_client.list_files = MagicMock(return_value=[])
        mock_client.complete = AsyncMock(return_value="This is not JSON")
        mock_client.write_file = AsyncMock()

        task = TaskModel(
            id="2",
            description="Implement feature X",
            files=["src/feature.py"],
        )
        agent = WorkerAgent(task=task, client=mock_client)

        result = await agent.run()

        lower = result.lower()
        assert "could not" in lower or "parse" in lower
        assert mock_client.write_file.call_count >= 1

    @pytest.mark.asyncio
    async def test_worker_reads_existing_files(self, tmp_path: Path) -> None:
        """read_file is awaited for each file listed in the task."""
        mock_client = AsyncMock()
        mock_client.settings.worker_model = "test-model"
        mock_client.settings.workspace = tmp_path
        mock_client.read_file = AsyncMock(return_value="existing content")
        mock_client.list_files = MagicMock(return_value=[])
        mock_client.complete = AsyncMock(
            return_value='{"files": [], "summary": "No changes needed"}'
        )
        mock_client.write_file = AsyncMock()

        task = TaskModel(
            id="3",
            description="Update existing files",
            files=["src/existing.py"],
        )
        agent = WorkerAgent(task=task, client=mock_client)

        result = await agent.run()

        assert mock_client.read_file.call_count == len(task.files)
        mock_client.read_file.assert_any_call("src/existing.py")
        assert "No changes needed" in result
