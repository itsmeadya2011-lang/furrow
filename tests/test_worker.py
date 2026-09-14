import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient


class TestWorkerAgentRun:
    @pytest.mark.asyncio
    async def test_run_reads_files_calls_llm_writes_files(self, tmp_path):
        input_file = tmp_path / "input.txt"
        input_file.write_text("source code")

        output_file = tmp_path / "output.txt"

        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(return_value="source code")
        mock_client.write_file = AsyncMock(return_value=None)
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "files": {str(output_file): "written content"},
            "summary": "task done",
        }))
        mock_client.settings.worker_model = "test-worker"

        task = TaskModel(id="1", description="write code", files=[str(input_file)])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        mock_client.read_file.assert_awaited_once_with(str(input_file))
        mock_client.complete.assert_awaited_once()
        mock_client.write_file.assert_awaited_once_with(str(output_file), "written content")
        assert result == "task done"
        assert output_file.exists()
        assert output_file.read_text() == "written content"

    @pytest.mark.asyncio
    async def test_run_with_no_files(self, tmp_path):
        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(side_effect=FileNotFoundError("no file"))
        mock_client.write_file = AsyncMock(return_value=None)
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "files": {},
            "summary": "no files task",
        }))
        mock_client.settings.worker_model = "test-worker"

        task = TaskModel(id="1", description="no files task", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "no files task"

    @pytest.mark.asyncio
    async def test_run_returns_raw_text_on_invalid_json(self, tmp_path):
        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(return_value="source")
        mock_client.write_file = AsyncMock(return_value=None)
        mock_client.complete = AsyncMock(return_value="this is not JSON")
        mock_client.settings.worker_model = "test-worker"

        task = TaskModel(id="1", description="bad response", files=[str(tmp_path / "f.txt")])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "this is not JSON"

    @pytest.mark.asyncio
    async def test_run_writes_multiple_files(self, tmp_path):
        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(return_value="source")
        mock_client.write_file = AsyncMock(return_value=None)
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "files": {
                str(tmp_path / "a.py"): "content a",
                str(tmp_path / "b.py"): "content b",
            },
            "summary": "multi file",
        }))
        mock_client.settings.worker_model = "test-worker"

        task = TaskModel(id="1", description="multi", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "multi file"
        assert (tmp_path / "a.py").read_text() == "content a"
        assert (tmp_path / "b.py").read_text() == "content b"
        assert mock_client.write_file.await_count == 2

    @pytest.mark.asyncio
    async def test_run_handles_file_not_found_gracefully(self, tmp_path):
        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(side_effect=FileNotFoundError("missing"))
        mock_client.write_file = AsyncMock(return_value=None)
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "files": {str(tmp_path / "out.txt"): "ok"},
            "summary": "recovered",
        }))
        mock_client.settings.worker_model = "test-worker"

        task = TaskModel(id="1", description="test", files=["/nonexistent/file.txt"])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "recovered"

    @pytest.mark.asyncio
    async def test_run_json_with_nested_file_paths(self, tmp_path):
        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(return_value="source")
        mock_client.write_file = AsyncMock(return_value=None)
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "files": {
                "subdir/nested.py": "nested content",
            },
            "summary": "nested",
        }))
        mock_client.settings.worker_model = "test-worker"

        task = TaskModel(id="1", description="nested", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "nested"
        written_path = tmp_path / "subdir" / "nested.py"
        assert written_path.exists()
        assert written_path.read_text() == "nested content"

    @pytest.mark.asyncio
    async def test_run_passes_file_contents_to_llm(self, tmp_path):
        input_file = tmp_path / "input.py"
        input_file.write_text("def hello(): pass")

        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(return_value="def hello(): pass")
        mock_client.write_file = AsyncMock(return_value=None)
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "files": {},
            "summary": "ok",
        }))
        mock_client.settings.worker_model = "test-worker"

        task = TaskModel(id="1", description="test", files=[str(input_file)])
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        mock_client.read_file.assert_awaited_once_with(str(input_file))
        call_args = mock_client.complete.call_args
        prompt = call_args.args[0]
        assert "def hello(): pass" in prompt
        assert str(input_file) in prompt

    @pytest.mark.asyncio
    async def test_run_empty_json_files(self, tmp_path):
        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(return_value="source")
        mock_client.write_file = AsyncMock(return_value=None)
        mock_client.complete = AsyncMock(return_value=json.dumps({
            "files": {},
            "summary": "empty files",
        }))
        mock_client.settings.worker_model = "test-worker"

        task = TaskModel(id="1", description="test", files=[])
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "empty files"
        mock_client.write_file.assert_not_awaited()
