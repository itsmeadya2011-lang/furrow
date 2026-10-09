from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import FILE_BLOCK_RE, WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


class TestWorkerAgent:
    @pytest.fixture
    def mock_client(self):
        client = MagicMock(spec=LLMClient)
        client.settings = Settings(workspace="/tmp/test_workspace", worker_model="test-model")
        client.complete = AsyncMock(return_value="Here is the code:\nFILE: test.py\n```python\nprint('hello')\n```")
        client.list_files = MagicMock(return_value=[])
        client.read_file = AsyncMock(return_value="")
        client.write_file = AsyncMock()
        return client

    @pytest.fixture
    def task(self):
        return TaskModel(id="1", description="Create a test file", files=["test.py"])

    @pytest.mark.asyncio
    async def test_run_parses_file_blocks_and_writes_files(self, mock_client, task):
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        mock_client.write_file.assert_called_once()
        call_args = mock_client.write_file.call_args
        assert "test.py" in str(call_args[0][0])
        assert "print('hello')" in call_args[0][1]
        assert "Wrote files: test.py" in result

    @pytest.mark.asyncio
    async def test_run_returns_summary_when_files_written(self, mock_client, task):
        mock_client.complete.return_value = (
            "FILE: main.py\n```python\ndef main():\n    pass\n```\n"
            "FILE: utils.py\n```python\ndef util():\n    pass\n```"
        )
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert mock_client.write_file.call_count == 2
        assert "Wrote files: main.py, utils.py" in result

    @pytest.mark.asyncio
    async def test_run_returns_raw_response_when_no_file_blocks(self, mock_client, task):
        mock_client.complete.return_value = "This is just a text response without any file blocks."
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert result == "This is just a text response without any file blocks."
        mock_client.write_file.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_with_multiple_file_blocks_same_file(self, mock_client, task):
        mock_client.complete.return_value = (
            "FILE: test.py\n```python\nprint('first')\n```\n"
            "FILE: test.py\n```python\nprint('second')\n```"
        )
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert mock_client.write_file.call_count == 2
        assert "Wrote files: test.py, test.py" in result

    @pytest.mark.asyncio
    async def test_run_handles_write_errors_gracefully(self, mock_client, task):
        mock_client.write_file.side_effect = Exception("Permission denied")
        worker = WorkerAgent(task=task, client=mock_client)
        result = await worker.run()

        assert "Wrote files" not in result
        assert "print('hello')" in result

    @pytest.mark.asyncio
    async def test_run_reads_existing_files_for_context(self, mock_client, task):
        mock_client.list_files.return_value = ["existing.py"]
        mock_client.read_file.return_value = "# existing content"
        mock_client.complete.return_value = "FILE: new.py\n```python\nprint('new')\n```"
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        mock_client.read_file.assert_called()

    @pytest.mark.asyncio
    async def test_run_prioritizes_task_files(self, mock_client, task):
        task.files = ["priority.py"]
        mock_client.list_files.return_value = ["priority.py", "other.py"]
        mock_client.complete.return_value = "FILE: new.py\n```python\nprint('new')\n```"
        worker = WorkerAgent(task=task, client=mock_client)
        await worker.run()

        # read_file is called with a Path object
        mock_client.read_file.assert_called()
        call_args = mock_client.read_file.call_args[0][0]
        assert str(call_args).endswith("priority.py")


class TestFileBlockRegex:
    def test_matches_simple_file_block(self):
        text = "FILE: test.py\n```python\nprint('hello')\n```"
        matches = list(FILE_BLOCK_RE.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1).strip() == "test.py"
        assert matches[0].group(3) == "print('hello')\n"

    def test_matches_file_block_without_language(self):
        text = "FILE: test.txt\n```\nplain text\n```"
        matches = list(FILE_BLOCK_RE.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1).strip() == "test.txt"
        assert matches[0].group(2) is None
        assert matches[0].group(3) == "plain text\n"

    def test_matches_multiple_file_blocks(self):
        text = (
            "FILE: a.py\n```python\n1\n```\n"
            "FILE: b.py\n```python\n2\n```"
        )
        matches = list(FILE_BLOCK_RE.finditer(text))
        assert len(matches) == 2
        assert matches[0].group(1).strip() == "a.py"
        assert matches[1].group(1).strip() == "b.py"

    def test_handles_whitespace_in_path(self):
        text = "FILE:  path/to/file.py  \n```python\ncode\n```"
        matches = list(FILE_BLOCK_RE.finditer(text))
        assert len(matches) == 1
        assert matches[0].group(1).strip() == "path/to/file.py"