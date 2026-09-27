"""Tests for WorkerAgent — file operation parsing and file writing."""
from __future__ import annotations

import asyncio

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import FileOperation, TaskModel
from furrow.llm import LLMClient

from tests.conftest import FakeLLMClient, fake_settings, fake_client  # noqa: F401


# --- _parse_file_operations ---


class TestParseFileOperations:
    def test_parse_single_write(self):
        response = '<write path="src/foo.py">print("hello")</write>'
        ops = WorkerAgent._parse_file_operations(response)
        assert len(ops) == 1
        assert ops[0].operation == "write"
        assert ops[0].path == "src/foo.py"
        assert ops[0].content == 'print("hello")'

    def test_parse_single_edit(self):
        response = (
            '<edit path="src/foo.py"><old>hello</old><new>world</new></edit>'
        )
        ops = WorkerAgent._parse_file_operations(response)
        assert len(ops) == 1
        assert ops[0].operation == "edit"
        assert ops[0].path == "src/foo.py"
        assert ops[0].old == "hello"
        assert ops[0].new == "world"

    def test_parse_multiple_operations(self):
        response = (
            '<write path="a.py">content_a</write>\n'
            '<edit path="b.py"><old>old_b</old><new>new_b</new></edit>\n'
            '<write path="c.py">content_c</write>'
        )
        ops = WorkerAgent._parse_file_operations(response)
        assert len(ops) == 3
        assert ops[0].operation == "write" and ops[0].path == "a.py"
        assert ops[1].operation == "edit" and ops[1].path == "b.py"
        assert ops[2].operation == "write" and ops[2].path == "c.py"

    def test_parse_multiline_content(self):
        response = '<write path="x.py">line1\nline2\nline3</write>'
        ops = WorkerAgent._parse_file_operations(response)
        assert len(ops) == 1
        assert ops[0].content == "line1\nline2\nline3"

    def test_parse_no_operations(self):
        response = "No file operations here, just a summary."
        ops = WorkerAgent._parse_file_operations(response)
        assert len(ops) == 0

    def test_parse_empty_response(self):
        ops = WorkerAgent._parse_file_operations("")
        assert len(ops) == 0

    def test_parse_edit_with_whitespace(self):
        response = (
            '<edit path="x.py">\n'
            '  <old>foo</old>\n'
            '  <new>bar</new>\n'
            '</edit>'
        )
        ops = WorkerAgent._parse_file_operations(response)
        assert len(ops) == 1
        assert ops[0].old == "foo"
        assert ops[0].new == "bar"

    def test_parse_unmatched_edit_without_old_new(self):
        """An <edit> block missing <old>/<new> should be skipped."""
        response = '<edit path="x.py">no old/new tags</edit>'
        ops = WorkerAgent._parse_file_operations(response)
        assert len(ops) == 0


# --- _apply_edit ---


class TestApplyEdit:
    async def test_apply_edit_replaces_first_occurrence(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("hello hello world")
        client = FakeLLMClient()
        worker = WorkerAgent(
            task=TaskModel(id="1", description="test"),
            client=client,
        )
        # Override the client's write_file / read_file to use the real filesystem
        import aiofiles

        async def real_write(path, content):
            p = __import__("pathlib").Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)

        async def real_read(path):
            async with aiofiles.open(path, "r") as f:
                return await f.read()

        client.write_file = real_write
        client.read_file = real_read

        await worker._apply_edit(str(filepath), "hello", "bye")
        content = filepath.read_text()
        assert content == "bye hello world"

    async def test_apply_edit_no_match_is_noop(self, tmp_path):
        filepath = tmp_path / "test.txt"
        filepath.write_text("original content")
        client = FakeLLMClient()

        import aiofiles
        from pathlib import Path

        async def real_write(path, content):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)

        async def real_read(path):
            async with aiofiles.open(path, "r") as f:
                return await f.read()

        client.write_file = real_write
        client.read_file = real_read

        worker = WorkerAgent(task=TaskModel(id="1", description="t"), client=client)
        await worker._apply_edit(str(filepath), "nonexistent", "replacement")
        content = filepath.read_text()
        assert content == "original content"


# --- WorkerAgent.run ---


class TestWorkerRun:
    async def test_run_writes_file_from_llm_response(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        client = FakeLLMClient()
        client._response = '<write path="output.txt">hello world</write>'

        import aiofiles
        from pathlib import Path

        async def real_write(path, content):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)

        client.write_file = real_write

        worker = WorkerAgent(
            task=TaskModel(id="1", description="create a file"),
            client=client,
        )
        result = await worker.run()

        assert "output.txt" in result
        assert (tmp_path / "output.txt").read_text() == "hello world"

    async def test_run_edits_existing_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        # Pre-create the file
        (tmp_path / "existing.py").write_text("old line")

        client = FakeLLMClient()
        client._response = (
            '<edit path="existing.py"><old>old line</old><new>new line</new></edit>'
        )

        import aiofiles
        from pathlib import Path

        async def real_write(path, content):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)

        async def real_read(path):
            async with aiofiles.open(path, "r") as f:
                return await f.read()

        client.write_file = real_write
        client.read_file = real_read

        worker = WorkerAgent(
            task=TaskModel(
                id="2",
                description="update existing file",
                files=["existing.py"],
            ),
            client=client,
        )
        result = await worker.run()

        assert "existing.py" in result
        assert (tmp_path / "existing.py").read_text() == "new line"

    async def test_run_reads_existing_file_as_context(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "target.py").write_text("original")

        client = FakeLLMClient()
        client._response = '<write path="target.py">updated</write>'

        original_complete = client.complete
        captured_prompts: list[str] = []

        async def capture_complete(prompt, system="", model=None):
            captured_prompts.append(prompt)
            return await original_complete(prompt, system, model)

        client.complete = capture_complete

        import aiofiles
        from pathlib import Path

        async def real_write(path, content):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)

        async def real_read(path):
            async with aiofiles.open(path, "r") as f:
                return await f.read()

        client.write_file = real_write
        client.read_file = real_read

        worker = WorkerAgent(
            task=TaskModel(
                id="3",
                description="update file",
                files=["target.py"],
            ),
            client=client,
        )
        await worker.run()

        # The prompt should contain the existing file content
        assert "original" in captured_prompts[0]
        assert "target.py" in captured_prompts[0]

    async def test_run_returns_response_when_no_operations(self):
        client = FakeLLMClient()
        client._response = "I would create a file but didn't output XML."

        worker = WorkerAgent(task=TaskModel(id="1", description="test"), client=client)
        result = await worker.run()
        assert result == "I would create a file but didn't output XML."

    async def test_run_handles_nonexistent_task_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        client = FakeLLMClient()
        # LLM returns an edit for a file that doesn't exist yet
        client._response = '<write path="new.py">print("hi")</write>'

        import aiofiles
        from pathlib import Path

        async def real_write(path, content):
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(p, "w") as f:
                await f.write(content)

        client.write_file = real_write

        async def real_read(path):
            if not Path(path).exists():
                raise FileNotFoundError(path)
            async with aiofiles.open(path, "r") as f:
                return await f.read()

        client.read_file = real_read

        worker = WorkerAgent(
            task=TaskModel(
                id="4",
                description="create new file",
                files=["new.py"],
            ),
            client=client,
        )
        result = await worker.run()
        assert "new.py" in result
        assert (tmp_path / "new.py").read_text() == 'print("hi")'
