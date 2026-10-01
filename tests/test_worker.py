"""Tests for the worker agent's file writing logic."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import FILE_BLOCK_PATTERN, WorkerAgent
from furrow.config import Settings, TaskModel


class TestFileBlockPattern:
    """Tests for the FILE: block regex pattern."""

    def test_single_file_block(self):
        response = (
            "Some preamble\nFILE: src/main.py\n```python\nprint('hello')\n```\nSome trailing text"
        )
        matches = FILE_BLOCK_PATTERN.findall(response)
        assert len(matches) == 1
        assert matches[0][0].strip() == "src/main.py"
        assert "print('hello')" in matches[0][1]

    def test_multiple_file_blocks(self):
        response = (
            "FILE: src/main.py\n```python\nx = 1\n```\nFILE: src/utils.py\n```python\ny = 2\n```\n"
        )
        matches = FILE_BLOCK_PATTERN.findall(response)
        assert len(matches) == 2
        assert matches[0][0].strip() == "src/main.py"
        assert matches[1][0].strip() == "src/utils.py"

    def test_no_file_blocks(self):
        response = "No files here, just text."
        matches = FILE_BLOCK_PATTERN.findall(response)
        assert len(matches) == 0

    def test_file_block_with_different_language(self):
        response = "FILE: README.md\n```markdown\n# Title\n```\n"
        matches = FILE_BLOCK_PATTERN.findall(response)
        assert len(matches) == 1

    def test_file_block_without_language(self):
        response = "FILE: config.yaml\n```\nkey: value\n```\n"
        matches = FILE_BLOCK_PATTERN.findall(response)
        assert len(matches) == 1

    def test_multiline_content(self):
        response = (
            "FILE: src/app.py\n"
            "```python\n"
            "def foo():\n"
            "    return {\n"
            "        'a': 1,\n"
            "        'b': 2,\n"
            "    }\n"
            "```\n"
        )
        matches = FILE_BLOCK_PATTERN.findall(response)
        assert len(matches) == 1
        content = matches[0][1]
        assert "def foo():" in content
        assert "return {" in content


class TestWorkerFileWriting:
    """Tests for _write_files_from_response."""

    @pytest.mark.asyncio
    async def test_write_files_to_disk(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test"),
            settings=settings,
        )
        worker.client = MagicMock()
        worker.client.settings = settings
        worker.client.write_file = AsyncMock()

        response = "FILE: src/main.py\n```python\nprint('hello')\n```\n"

        result = await worker._write_files_from_response(response)
        assert result == ["src/main.py"]
        worker.client.write_file.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_write_files_no_matches(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test"),
            settings=settings,
        )
        worker.client = MagicMock()
        worker.client.settings = settings
        worker.client.write_file = AsyncMock()

        result = await worker._write_files_from_response("No files here.")
        assert result == []
        worker.client.write_file.assert_not_awaited()


class TestWorkerResolvePath:
    """Tests for _resolve_path."""

    def test_relative_path(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test"),
            settings=settings,
        )
        path = worker._resolve_path("src/main.py")
        assert path == tmp_path / "src" / "main.py"

    def test_absolute_path(self, tmp_path):
        settings = Settings(workspace=tmp_path)
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test"),
            settings=settings,
        )
        path = worker._resolve_path("/usr/local/file.py")
        assert path == Path("/usr/local/file.py")


class TestWorkerRun:
    """Tests for the worker's run() method with mocked LLM."""

    @pytest.mark.asyncio
    async def test_worker_reads_existing_files(self, tmp_path):
        """Worker should read existing files for context."""
        settings = Settings(workspace=tmp_path)
        existing_file = tmp_path / "src" / "module.py"
        existing_file.parent.mkdir(parents=True)
        existing_file.write_text("existing content")

        worker = WorkerAgent(
            task=TaskModel(id="1", description="test", files=["src/module.py"]),
            settings=settings,
        )

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.read_file = AsyncMock(return_value="existing content")
        mock_client.write_file = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value="Completed. FILE: src/module.py\n```python\nnew content\n```"
        )
        worker.client = mock_client

        result = await worker.run()

        mock_client.read_file.assert_awaited_once()
        mock_client.write_file.assert_awaited_once()
        assert "src/module.py" in result

    @pytest.mark.asyncio
    async def test_worker_handles_missing_file(self, tmp_path):
        """Worker should handle files that don't exist yet."""
        settings = Settings(workspace=tmp_path)
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test", files=["src/new.py"]),
            settings=settings,
        )

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.read_file = AsyncMock(side_effect=FileNotFoundError("not found"))
        mock_client.write_file = AsyncMock()
        mock_client.complete = AsyncMock(
            return_value="Done. FILE: src/new.py\n```python\ncode\n```"
        )
        worker.client = mock_client

        result = await worker.run()

        assert "src/new.py" in result

    @pytest.mark.asyncio
    async def test_worker_llm_error_propagates(self, tmp_path):
        """LLM errors should propagate up."""
        settings = Settings(workspace=tmp_path)
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test"),
            settings=settings,
        )

        mock_client = MagicMock()
        mock_client.settings = settings
        mock_client.read_file = AsyncMock(return_value="")
        mock_client.complete = AsyncMock(side_effect=RuntimeError("LLM down"))
        worker.client = mock_client

        with pytest.raises(RuntimeError, match="LLM down"):
            await worker.run()
