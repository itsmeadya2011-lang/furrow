import pytest
from unittest.mock import AsyncMock
from pathlib import Path

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class TestWorkerAgent:
    @pytest.mark.asyncio
    async def test_run_reads_existing_files_before_llm(self, tmp_path):
        test_file = tmp_path / "test.py"
        test_file.write_text("print('hello')")
        task = TaskModel(id="1", description="test", files=[str(test_file)])
        mock_client = AsyncMock()
        mock_client.read_file = AsyncMock(return_value="print('hello')")
        mock_client.complete = AsyncMock(
            return_value='{"files": {"test.py": "x=1"}, "summary": "ok"}'
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        mock_client.read_file.assert_called_once_with(str(test_file))

    @pytest.mark.asyncio
    async def test_run_writes_files_returned_by_llm(self, tmp_path):
        task = TaskModel(id="1", description="test", files=["test.py"])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"files": {"test.py": "x=1"}, "summary": "ok"}'
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        mock_client.write_file.assert_called_once_with("test.py", "x=1")

    @pytest.mark.asyncio
    async def test_run_handles_json_parse_error_gracefully(self, tmp_path):
        task = TaskModel(id="1", description="test", files=["test.py"])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(return_value="not valid json at all")

        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert "Failed to parse" in result

    @pytest.mark.asyncio
    async def test_run_handles_json_code_block_fallback(self, tmp_path):
        task = TaskModel(id="1", description="test", files=["test.py"])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value="```json\n{\"files\": {}, \"summary\": \"ok\"}\n```"
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert "ok" in result or "Completed" in result

    @pytest.mark.asyncio
    async def test_run_reads_multiple_files(self, tmp_path):
        file_a = tmp_path / "a.py"
        file_a.write_text("a=1")
        file_b = tmp_path / "b.py"
        file_b.write_text("b=2")
        task = TaskModel(id="1", description="test", files=[str(file_a), str(file_b)])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"files": {"a.py": "x=1"}, "summary": "ok"}'
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        assert mock_client.read_file.call_count == 2

    @pytest.mark.asyncio
    async def test_run_writes_multiple_files(self, tmp_path):
        task = TaskModel(id="1", description="test", files=["a.py", "b.py"])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"files": {"a.py": "x=1", "b.py": "y=2"}, "summary": "ok"}'
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        assert mock_client.write_file.call_count == 2

    @pytest.mark.asyncio
    async def test_run_handles_file_not_found_gracefully(self, tmp_path):
        missing_file = tmp_path / "missing.py"
        task = TaskModel(id="1", description="test", files=[str(missing_file)])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"files": {}, "summary": "ok"}'
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert "Completed" in result

    @pytest.mark.asyncio
    async def test_run_no_files_skips_read(self):
        task = TaskModel(id="1", description="test", files=[])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"files": {}, "summary": "ok"}'
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        mock_client.read_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_returns_summary_string(self):
        task = TaskModel(id="1", description="test", files=[])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value='{"files": {"f.py": "x=1"}, "summary": "added f"}'
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_run_handles_json_decode_error_in_code_block(self):
        task = TaskModel(id="1", description="test", files=[])
        mock_client = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value="```json\ninvalid json!!```"
        )
        mock_client.settings.worker_model = "test-model"

        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert "Failed to parse" in result
