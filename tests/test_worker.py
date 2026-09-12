import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class MockLLMClient:
    """A mock LLM client for testing."""

    def __init__(self, responses=None):
        self.settings = Settings()
        self.responses = responses or []
        self.written_files = {}
        self.read_files = {}

    async def complete(self, prompt, system="", model=None):
        if self.responses:
            return self.responses.pop(0)
        return "{}"

    async def read_file(self, path):
        return self.read_files.get(str(path), "")

    async def write_file(self, path, content):
        self.written_files[str(path)] = content

    def list_files(self, directory):
        return []


async def test_worker_run_with_files(tmp_path):
    client = MockLLMClient()
    client.settings.workspace = tmp_path
    client.responses = [
        json.dumps(
            {
                "files": {"output.py": "print('hello')"},
                "summary": "Created output.py",
                "issues": [],
            }
        )
    ]

    task = TaskModel(id="1", description="Create output.py", files=["output.py"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert "Created output.py" in result
    assert str(tmp_path / "output.py") in client.written_files
    assert client.written_files[str(tmp_path / "output.py")] == "print('hello')"


async def test_worker_run_no_files():
    client = MockLLMClient()
    client.responses = [
        json.dumps(
            {
                "files": {},
                "summary": "No files needed",
                "issues": [],
            }
        )
    ]

    task = TaskModel(id="1", description="Do something", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert "No files needed" in result


async def test_worker_run_invalid_json():
    client = MockLLMClient()
    client.responses = ["This is not valid JSON"]

    task = TaskModel(id="1", description="Do something", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "This is not valid JSON"


async def test_worker_run_disabled_edits(tmp_path):
    client = MockLLMClient()
    client.settings.workspace = tmp_path
    client.settings.enable_file_edits = False
    client.responses = [
        json.dumps(
            {
                "files": {"output.py": "print('hello')"},
                "summary": "Would create output.py",
                "issues": [],
            }
        )
    ]

    task = TaskModel(id="1", description="Create output.py", files=["output.py"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert "Would create output.py" in result
    # File should NOT be written
    assert len(client.written_files) == 0


async def test_worker_run_with_issues(tmp_path):
    client = MockLLMClient()
    client.settings.workspace = tmp_path
    client.responses = [
        json.dumps(
            {
                "files": {},
                "summary": "Could not complete",
                "issues": ["Missing dependency: foo"],
            }
        )
    ]

    task = TaskModel(id="1", description="Do something", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert "Could not complete" in result
    assert "Missing dependency: foo" in result