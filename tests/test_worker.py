import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from furrow.agents.worker import WorkerAgent, _extract_json
from furrow.config import TaskModel
from furrow.llm import LLMClient


class TestExtractJson:
    """Tests for the _extract_json helper function."""

    def test_extract_simple_json(self):
        """Test extracting simple JSON from response."""
        response = '{"summary": "done", "files": []}'
        result = _extract_json(response)
        assert result == {"summary": "done", "files": []}

    def test_extract_json_with_markdown_fences(self):
        """Test extracting JSON wrapped in markdown code fences."""
        response = '```json\n{"summary": "done", "files": []}\n```'
        result = _extract_json(response)
        assert result == {"summary": "done", "files": []}

    def test_extract_json_with_surrounding_text(self):
        """Test extracting JSON with surrounding text."""
        response = 'Here is the result:\n{"summary": "done", "files": []}\nDone!'
        result = _extract_json(response)
        assert result == {"summary": "done", "files": []}

    def test_extract_json_with_files(self):
        """Test extracting JSON with files array."""
        response = json.dumps({
            "summary": "Created auth module",
            "files": [
                {"path": "src/auth.py", "content": "def auth(): pass"},
                {"path": "tests/test_auth.py", "content": "def test_auth(): pass"},
            ],
        })
        result = _extract_json(response)
        assert result["summary"] == "Created auth module"
        assert len(result["files"]) == 2
        assert result["files"][0]["path"] == "src/auth.py"

    def test_extract_json_raises_on_invalid(self):
        """Test raising on completely invalid JSON."""
        response = "This is not JSON at all"
        with pytest.raises(json.JSONDecodeError):
            _extract_json(response)

    def test_extract_json_raises_on_non_object(self):
        """Test raising when JSON is not an object."""
        response = '[1, 2, 3]'
        with pytest.raises(ValueError, match="Expected JSON object"):
            _extract_json(response)


class TestWorkerAgent:
    """Tests for the WorkerAgent class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock LLMClient."""
        client = MagicMock(spec=LLMClient)
        client.settings = MagicMock()
        client.settings.workspace = Path("/tmp/test_workspace")
        client.settings.worker_model = "test-model"
        client.complete = AsyncMock()
        client.write_file = AsyncMock()
        return client

    @pytest.fixture
    def sample_task(self):
        """Create a sample task."""
        return TaskModel(
            id="1",
            description="Implement authentication",
            files=["src/auth.py"],
        )

    @pytest.mark.asyncio
    async def test_worker_parses_json_and_extracts_files(self, mock_client, sample_task):
        """Test worker parses JSON response and extracts files."""
        response = json.dumps({
            "summary": "Implemented auth module",
            "files": [
                {"path": "src/auth.py", "content": "def auth(): return True"},
            ],
        })
        mock_client.complete.return_value = response

        worker = WorkerAgent(task=sample_task, client=mock_client)
        result = await worker.run()

        assert "Implemented auth module" in result
        assert "wrote 1 file(s)" in result

    @pytest.mark.asyncio
    async def test_worker_calls_write_file_for_each_file(self, mock_client, sample_task):
        """Test worker calls write_file for each file in response."""
        response = json.dumps({
            "summary": "Created multiple files",
            "files": [
                {"path": "src/auth.py", "content": "def auth(): pass"},
                {"path": "tests/test_auth.py", "content": "def test_auth(): pass"},
            ],
        })
        mock_client.complete.return_value = response

        worker = WorkerAgent(task=sample_task, client=mock_client)
        await worker.run()

        assert mock_client.write_file.call_count == 2
        mock_client.write_file.assert_any_call(
            Path("/tmp/test_workspace/src/auth.py"),
            "def auth(): pass",
        )
        mock_client.write_file.assert_any_call(
            Path("/tmp/test_workspace/tests/test_auth.py"),
            "def test_auth(): pass",
        )

    @pytest.mark.asyncio
    async def test_worker_handles_malformed_json_gracefully(self, mock_client, sample_task):
        """Test worker handles malformed JSON gracefully."""
        mock_client.complete.return_value = "This is not JSON at all"

        worker = WorkerAgent(task=sample_task, client=mock_client)
        result = await worker.run()

        assert "Worker produced no parseable output" in result
        mock_client.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_worker_returns_summary_from_response(self, mock_client, sample_task):
        """Test worker returns summary from response."""
        response = json.dumps({
            "summary": "Fixed the bug in auth module",
            "files": [],
        })
        mock_client.complete.return_value = response

        worker = WorkerAgent(task=sample_task, client=mock_client)
        result = await worker.run()

        assert "Fixed the bug in auth module" in result

    @pytest.mark.asyncio
    async def test_worker_handles_missing_summary(self, mock_client, sample_task):
        """Test worker handles response with missing summary."""
        response = json.dumps({"files": []})
        mock_client.complete.return_value = response

        worker = WorkerAgent(task=sample_task, client=mock_client)
        result = await worker.run()

        assert "Worker completed without a summary" in result

    @pytest.mark.asyncio
    async def test_worker_skips_invalid_file_entries(self, mock_client, sample_task):
        """Test worker skips invalid file entries."""
        response = json.dumps({
            "summary": "Mixed valid and invalid",
            "files": [
                {"path": "valid.py", "content": "valid content"},
                "not a dict",
                {"content": "missing path"},
                {"path": 123, "content": "invalid path type"},
                {"path": "valid2.py", "content": 123},
            ],
        })
        mock_client.complete.return_value = response

        worker = WorkerAgent(task=sample_task, client=mock_client)
        await worker.run()

        # Only the first valid file should be written
        assert mock_client.write_file.call_count == 1
        mock_client.write_file.assert_called_with(
            Path("/tmp/test_workspace/valid.py"),
            "valid content",
        )

    @pytest.mark.asyncio
    async def test_worker_handles_non_list_files(self, mock_client, sample_task):
        """Test worker handles when files is not a list."""
        response = json.dumps({
            "summary": "Files is not a list",
            "files": "not a list",
        })
        mock_client.complete.return_value = response

        worker = WorkerAgent(task=sample_task, client=mock_client)
        result = await worker.run()

        assert mock_client.write_file.call_count == 0
        assert "wrote 0 file(s)" in result
