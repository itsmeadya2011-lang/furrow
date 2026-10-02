"""Tests for the WorkerAgent."""
import json
from pathlib import Path

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from tests.conftest import MockLLMClient


class TestParseResponse:
    def test_plain_json(self):
        response = json.dumps({"changes": [], "summary": "done"})
        result = WorkerAgent._parse_response(response)
        assert result is not None
        assert result["summary"] == "done"
        assert result["changes"] == []

    def test_markdown_fenced_json(self):
        response = '```json\n{"changes": [{"path": "a.py", "content": "x=1"}], "summary": "ok"}\n```'
        result = WorkerAgent._parse_response(response)
        assert result is not None
        assert len(result["changes"]) == 1
        assert result["changes"][0]["path"] == "a.py"

    def test_json_with_surrounding_text(self):
        response = 'Here is the plan:\n{"changes": [], "summary": "ok"}\nHope you like it.'
        result = WorkerAgent._parse_response(response)
        assert result is not None
        assert result["summary"] == "ok"

    def test_invalid_json_returns_none(self):
        result = WorkerAgent._parse_response("this is not json at all")
        assert result is None

    def test_empty_string(self):
        result = WorkerAgent._parse_response("")
        assert result is None

    def test_fence_without_language(self):
        response = '```\n{"changes": [], "summary": "test"}\n```'
        result = WorkerAgent._parse_response(response)
        assert result is not None
        assert result["summary"] == "test"


class TestWorkerRun:
    @pytest.mark.asyncio
    async def test_writes_files_from_changes(self):
        response = json.dumps({
            "changes": [
                {"path": "new_file.py", "content": "print('hello')"},
                {"path": "tests/test_new.py", "content": "def test_basic(): pass"},
            ],
            "summary": "Created two files",
        })
        client = MockLLMClient(responses=[response])
        task = TaskModel(id="1", description="Create hello module", files=["new_file.py"])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert "Created two files" in result
        assert any("new_file.py" in k for k in client.written_files)
        assert client.written_files["new_file.py"] == "print('hello')"

    @pytest.mark.asyncio
    async def test_handles_invalid_json_response(self):
        client = MockLLMClient(responses=["not valid json"])
        task = TaskModel(id="1", description="Some task", files=[])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert "Failed to parse" in result

    @pytest.mark.asyncio
    async def test_reads_existing_files(self):
        client = MockLLMClient(existing_files={"src/app.py": "def main(): pass"})
        client._responses = [json.dumps({"changes": [], "summary": "ok"})]
        task = TaskModel(id="1", description="Modify app", files=["src/app.py"])
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        # The file should have been read (we can check the prompt included content)
        assert "def main(): pass" in client.complete_calls[0][0]

    @pytest.mark.asyncio
    async def test_empty_changes(self):
        response = json.dumps({"changes": [], "summary": "No changes needed"})
        client = MockLLMClient(responses=[response])
        task = TaskModel(id="1", description="Review code", files=[])
        worker = WorkerAgent(task=task, client=client)
        result = await worker.run()
        assert "No changes needed" in result
        assert "none" in result.lower()

    @pytest.mark.asyncio
    async def test_no_file_path_skipped(self):
        response = json.dumps({
            "changes": [{"content": "x=1"}],  # missing path
            "summary": "ok",
        })
        client = MockLLMClient(responses=[response])
        task = TaskModel(id="1", description="test", files=[])
        worker = WorkerAgent(task=task, client=client)
        await worker.run()
        assert len(client.written_files) == 0


class TestGatherFiles:
    @pytest.mark.asyncio
    async def test_reads_specific_files(self):
        client = MockLLMClient(
            existing_files={"src/module.py": "class Foo: pass", "src/utils.py": "def bar(): pass"}
        )
        task = TaskModel(id="1", description="test", files=["src/module.py", "src/utils.py"])
        worker = WorkerAgent(task=task, client=client)
        result = await worker._gather_files(Path("/tmp/furrow_test_workspace"))
        assert "class Foo" in result
        assert "def bar" in result

    @pytest.mark.asyncio
    async def test_handles_missing_file(self):
        client = MockLLMClient()
        task = TaskModel(id="1", description="test", files=["nonexistent.py"])
        worker = WorkerAgent(task=task, client=client)
        result = await worker._gather_files(Path("/tmp/test"))
        assert "does not exist yet" in result

    @pytest.mark.asyncio
    async def test_no_files_lists_workspace(self):
        client = MockLLMClient(
            existing_files={"src/a.py": "x", "src/b.py": "y"}
        )
        task = TaskModel(id="1", description="test", files=[])
        worker = WorkerAgent(task=task, client=client)
        result = await worker._gather_files(Path("/tmp/test"))
        assert "workspace" in result.lower() or "file" in result.lower()
