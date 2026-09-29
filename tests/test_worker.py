"""Unit tests for the WorkerAgent."""

import json

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel, WorkerResult


class FakeLLMClient:
    """Minimal LLM client stub that returns canned responses."""

    def __init__(self, response: str, settings: Settings | None = None):
        self._response = response
        self._settings = settings or Settings()
        self.written_files: dict[str, str] = {}

    @property
    def settings(self):
        return self._settings

    async def complete(self, prompt, system="", model=None, timeout=None):
        return self._response

    async def read_file(self, path):
        return "existing content"

    async def write_file(self, path, content):
        self.written_files[str(path)] = content

    def list_files(self, directory):
        return []


@pytest.mark.asyncio
async def test_worker_parses_valid_json(tmp_path):
    """Worker should parse valid JSON and write files."""
    response = json.dumps({
        "summary": "Created new module",
        "files": [
            {"path": "src/new.py", "content": "def hello():\n    return 'world'"},
        ],
    })
    client = FakeLLMClient(response, settings=Settings(workspace=tmp_path))
    task = TaskModel(id="1", description="Create new module", files=["src/new.py"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert isinstance(result, WorkerResult)
    assert result.summary == "Created new module"
    assert len(result.files) == 1
    assert result.files[0].path == "src/new.py"
    # File should have been written.
    expected_path = str(tmp_path / "src" / "new.py")
    assert expected_path in client.written_files
    assert client.written_files[expected_path] == "def hello():\n    return 'world'"


@pytest.mark.asyncio
async def test_worker_handles_invalid_json(tmp_path):
    """Worker should return a WorkerResult with empty files on parse failure."""
    client = FakeLLMClient("not valid json{{{", settings=Settings(workspace=tmp_path))
    task = TaskModel(id="1", description="Do something")
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert isinstance(result, WorkerResult)
    assert result.files == []
    assert "Failed to parse" in result.summary


@pytest.mark.asyncio
async def test_worker_no_files_modified(tmp_path):
    """Worker with no files in response should not write anything."""
    response = json.dumps({"summary": "Nothing to do", "files": []})
    client = FakeLLMClient(response, settings=Settings(workspace=tmp_path))
    task = TaskModel(id="1", description="Investigate")
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert result.summary == "Nothing to do"
    assert result.files == []
    assert client.written_files == {}


@pytest.mark.asyncio
async def test_worker_reads_existing_files(tmp_path):
    """Worker should include existing file contents in the prompt."""
    existing = tmp_path / "existing.py"
    existing.write_text("x = 1\n")
    response = json.dumps({
        "summary": "Updated",
        "files": [
            {"path": "existing.py", "content": "x = 2\n"},
        ],
    })
    client = FakeLLMClient(response, settings=Settings(workspace=tmp_path))
    task = TaskModel(id="1", description="Update x", files=["existing.py"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert result.summary == "Updated"
    # File should be written with new content.
    expected_path = str(tmp_path / "existing.py")
    assert expected_path in client.written_files
    assert client.written_files[expected_path] == "x = 2\n"
