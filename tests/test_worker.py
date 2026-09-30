"""Tests for furrow.agents.worker — JSON parsing, file editing, and error handling."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient


def _make_worker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    client = LLMClient()
    worker = WorkerAgent(task=TaskModel(id="1", description="test task"), client=client)
    return worker


async def test_worker_parses_json_and_writes_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    worker = _make_worker(tmp_path, monkeypatch)

    fake_response = json.dumps({
        "edits": [
            {"path": "src/main.py", "content": "def main():\n    pass"},
            {"path": "tests/test_main.py", "content": "def test_main():\n    assert True"},
        ],
        "summary": "Created main module and tests",
    })

    with patch.object(worker.client, "complete", new=AsyncMock(return_value=fake_response)):
        result = await worker.run()

    assert result == "Created main module and tests"
    assert (tmp_path / "src" / "main.py").read_text() == "def main():\n    pass"
    assert (tmp_path / "tests" / "test_main.py").read_text() == "def test_main():\n    assert True"


async def test_worker_handles_no_edits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    worker = _make_worker(tmp_path, monkeypatch)
    fake_response = json.dumps({"edits": [], "summary": "No changes needed"})

    with patch.object(worker.client, "complete", new=AsyncMock(return_value=fake_response)):
        result = await worker.run()

    assert result == "No changes needed"


async def test_worker_handles_markdown_fenced_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    worker = _make_worker(tmp_path, monkeypatch)
    fake_response = "```json\n" + json.dumps({
        "edits": [
            {"path": "app.py", "content": "print('hello')"},
        ],
        "summary": "Created app.py",
    }) + "\n```"

    with patch.object(worker.client, "complete", new=AsyncMock(return_value=fake_response)):
        result = await worker.run()

    assert result == "Created app.py"
    assert (tmp_path / "app.py").read_text() == "print('hello')"


async def test_worker_invalid_json_returns_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    worker = _make_worker(tmp_path, monkeypatch)
    fake_response = "This is not JSON at all"

    with patch.object(worker.client, "complete", new=AsyncMock(return_value=fake_response)):
        result = await worker.run()

    assert "Could not parse" in result


async def test_worker_reads_existing_files_for_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Worker should read existing files to provide context to the LLM."""
    worker = _make_worker(tmp_path, monkeypatch)
    (tmp_path / "existing.py").write_text("# old content")

    worker.task = TaskModel(
        id="1", description="update file", files=["existing.py"]
    )

    captured_prompt = None

    async def capture_complete(prompt: str, **kwargs) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps({"edits": [], "summary": "done"})

    with patch.object(worker.client, "complete", new=capture_complete):
        await worker.run()

    assert captured_prompt is not None
    assert "existing.py" in captured_prompt
    assert "# old content" in captured_prompt


async def test_worker_missing_file_in_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Worker should handle files in task.files that don't exist yet."""
    worker = _make_worker(tmp_path, monkeypatch)
    worker.task = TaskModel(
        id="1", description="create file", files=["new_file.py"]
    )

    captured_prompt = None

    async def capture_complete(prompt: str, **kwargs) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps({"edits": [], "summary": "done"})

    with patch.object(worker.client, "complete", new=capture_complete):
        await worker.run()

    assert "new_file.py" in captured_prompt
    assert "does not exist" in captured_prompt


async def test_worker_truncates_long_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Summaries longer than _MAX_SUMMARY_LEN should be truncated."""
    worker = _make_worker(tmp_path, monkeypatch)
    long_summary = "x" * 5000
    fake_response = json.dumps({
        "edits": [{"path": "a.py", "content": "x"}],
        "summary": long_summary,
    })

    with patch.object(worker.client, "complete", new=AsyncMock(return_value=fake_response)):
        result = await worker.run()

    assert len(result) <= 2000
