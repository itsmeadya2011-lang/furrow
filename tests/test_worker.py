import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class TestWorkerAgent:
    @pytest.fixture
    def mock_client(self):
        settings = Settings(anthropic_api_key="test-key", workspace=Path("/tmp/workspace"))
        client = LLMClient(settings=settings)
        client.complete = AsyncMock()
        client.read_file = AsyncMock()
        client.write_file = AsyncMock()
        client.list_files = MagicMock(return_value=["existing.py"])
        return client

    @pytest.fixture
    def task(self):
        return TaskModel(
            id="1",
            description="Create a test file",
            files=["test_new.py"],
            dependencies=[],
        )

    @pytest.mark.asyncio
    async def test_run_with_mocked_llm_and_filesystem(self, mock_client, task, tmp_path):
        task.files = ["new_file.py"]
        response_data = {
            "changed_files": [{"path": "new_file.py", "content": "print('hello')"}],
            "summary": "Created new file",
            "issues": [],
        }
        mock_client.complete.return_value = json.dumps(response_data)
        mock_client.settings.workspace = tmp_path

        worker = WorkerAgent(task=task, client=mock_client, max_retries=1)
        result = await worker.run()

        assert "Created new file" in result
        mock_client.write_file.assert_called_once()

    @pytest.mark.asyncio
    async def test_json_response_parsing_markdown_fences(self, mock_client, task):
        response = """```json
        {
            "changed_files": [{"path": "test.py", "content": "code"}],
            "summary": "Done",
            "issues": []
        }
        ```"""
        mock_client.complete.return_value = response

        worker = WorkerAgent(task=task, client=mock_client, max_retries=1)
        result = await worker._get_llm_response_with_retry("prompt")

        assert result is not None
        assert result["changed_files"][0]["path"] == "test.py"

    @pytest.mark.asyncio
    async def test_json_response_parsing_plain_json(self, mock_client, task):
        response = '{"changed_files": [{"path": "test.py", "content": "code"}], "summary": "Done", "issues": []}'
        mock_client.complete.return_value = response

        worker = WorkerAgent(task=task, client=mock_client, max_retries=1)
        result = await worker._get_llm_response_with_retry("prompt")

        assert result is not None
        assert result["changed_files"][0]["path"] == "test.py"

    @pytest.mark.asyncio
    async def test_parse_response_various_formats(self, task):
        worker = WorkerAgent(task=task)

        # Test with markdown fences
        parsed = worker._parse_response('```json\n{"changed_files": [], "summary": "ok"}\n```')
        assert parsed["summary"] == "ok"

        # Test with generic markdown fences
        parsed = worker._parse_response('```\n{"changed_files": [], "summary": "ok"}\n```')
        assert parsed["summary"] == "ok"

        # Test with JSON embedded in text
        parsed = worker._parse_response('Some text\n{"changed_files": [], "summary": "embedded"}\nmore text')
        assert parsed["summary"] == "embedded"

        # Test string items in changed_files
        parsed = worker._parse_response('{"changed_files": ["file1.py", "file2.py"], "summary": "test"}')
        assert len(parsed["changed_files"]) == 2
        assert parsed["changed_files"][0] == {"path": "file1.py", "content": ""}
        assert parsed["changed_files"][1] == {"path": "file2.py", "content": ""}

    @pytest.mark.asyncio
    async def test_parse_response_invalid_raises_error(self, task):
        worker = WorkerAgent(task=task)

        with pytest.raises(ValueError, match="Expecting value"):
            worker._parse_response("not json")

        with pytest.raises(ValueError, match="changed_files must be a list"):
            worker._parse_response('{"changed_files": "not a list"}')

    @pytest.mark.asyncio
    async def test_apply_changes_writes_files_correctly(self, task, tmp_path):
        settings = Settings(anthropic_api_key="test-key", workspace=tmp_path)
        client = LLMClient(settings=settings)
        worker = WorkerAgent(task=task, client=client)

        result = {
            "changed_files": [
                {"path": "file1.py", "content": "content1"},
                {"path": "subdir/file2.py", "content": "content2"},
            ],
            "summary": "Done",
            "issues": [],
        }

        issues = await worker._apply_changes(result)

        assert issues == []
        assert (tmp_path / "file1.py").read_text() == "content1"
        assert (tmp_path / "subdir" / "file2.py").read_text() == "content2"

    @pytest.mark.asyncio
    async def test_apply_changes_handles_write_errors(self, mock_client, task, tmp_path):
        mock_client.settings.workspace = tmp_path
        mock_client.write_file.side_effect = PermissionError("Permission denied")
        worker = WorkerAgent(task=task, client=mock_client)

        result = {"changed_files": [{"path": "file1.py", "content": "content"}], "summary": "Done", "issues": []}
        issues = await worker._apply_changes(result)

        assert len(issues) == 1
        assert "Permission denied" in issues[0]

    @pytest.mark.asyncio
    async def test_workspace_context_gathering(self, mock_client, task):
        mock_client.list_files.return_value = ["file1.py", "file2.py", "dir/file3.py"]

        worker = WorkerAgent(task=task, client=mock_client)
        files = worker._get_workspace_context()

        assert "file1.py" in files
        assert "dir/file3.py" in files
        mock_client.list_files.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_on_parse_failure(self, mock_client, task):
        mock_client.complete.side_effect = [
            "invalid json",
            '{"changed_files": [], "summary": "success on retry"}',
        ]

        worker = WorkerAgent(task=task, client=mock_client, max_retries=2)
        result = await worker._get_llm_response_with_retry("prompt")

        assert result is not None
        assert result["summary"] == "success on retry"
        assert mock_client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_max_retries_exhausted_returns_none(self, mock_client, task):
        mock_client.complete.return_value = "always invalid"

        worker = WorkerAgent(task=task, client=mock_client, max_retries=2)
        result = await worker._get_llm_response_with_retry("prompt")

        assert result is None
        assert mock_client.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_get_relevant_file_contents(self, mock_client, task, tmp_path):
        task.files = ["file1.py", "file2.py"]
        (tmp_path / "file1.py").write_text("content1")
        (tmp_path / "file2.py").write_text("content2")
        mock_client.settings.workspace = tmp_path
        mock_client.read_file.side_effect = lambda p: (tmp_path / p.name).read_text()

        worker = WorkerAgent(task=task, client=mock_client)
        contents = await worker._get_relevant_file_contents()

        assert "content1" in contents
        assert "content2" in contents
        assert "file1.py" in contents
        assert "file2.py" in contents