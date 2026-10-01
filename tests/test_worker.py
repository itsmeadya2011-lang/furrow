from __future__ import annotations

import json

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from tests.conftest import MockLLMClient


def _worker(responses: list[str], tmp_path) -> WorkerAgent:
    client = MockLLMClient(responses=responses)
    from furrow import config

    config.settings.workspace = tmp_path
    return WorkerAgent(task=TaskModel(id="1", description="implement x"), client=client)


class TestWorkerJSON:
    async def test_valid_task_result(self, tmp_path):
        resp = json.dumps(
            {
                "summary": "implemented",
                "edits": [{"file": "a.py", "operation": "create", "content": "print(1)"}],
                "issues": [],
            }
        )
        worker = _worker([resp], tmp_path)
        result = await worker.run()
        assert result == "implemented"

    async def test_invalid_json_returns_raw(self, tmp_path):
        worker = _worker(["not json at all"], tmp_path)
        result = await worker.run()
        assert result == "not json at all"

    async def test_empty_edits(self, tmp_path):
        resp = json.dumps({"summary": "no changes", "edits": [], "issues": []})
        worker = _worker([resp], tmp_path)
        result = await worker.run()
        assert result == "no changes"

    async def test_issues_appended_to_summary(self, tmp_path):
        resp = json.dumps({"summary": "done", "edits": [], "issues": ["warn one", "warn two"]})
        worker = _worker([resp], tmp_path)
        result = await worker.run()
        assert "done" in result
        assert "warn one" in result
        assert "warn two" in result


class TestWorkerFileEdits:
    async def test_create_file(self, tmp_path):
        resp = json.dumps(
            {
                "summary": "created",
                "edits": [{"file": str(tmp_path / "new.txt"), "operation": "create", "content": "hello"}],
                "issues": [],
            }
        )
        worker = _worker([resp], tmp_path)
        await worker.run()
        assert (tmp_path / "new.txt").read_text() == "hello"

    async def test_edit_file(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("old content")
        resp = json.dumps(
            {
                "summary": "edited",
                "edits": [
                    {
                        "file": str(target),
                        "operation": "edit",
                        "old": "old content",
                        "new": "new content",
                    }
                ],
                "issues": [],
            }
        )
        worker = _worker([resp], tmp_path)
        await worker.run()
        assert target.read_text() == "new content"

    async def test_delete_file(self, tmp_path):
        target = tmp_path / "del.txt"
        target.write_text("bye")
        resp = json.dumps(
            {
                "summary": "deleted",
                "edits": [{"file": str(target), "operation": "delete"}],
                "issues": [],
            }
        )
        worker = _worker([resp], tmp_path)
        await worker.run()
        assert not target.exists()

    async def test_failed_edit_records_issue(self, tmp_path):
        target = tmp_path / "file.txt"
        target.write_text("only this")
        resp = json.dumps(
            {
                "summary": "done",
                "edits": [
                    {
                        "file": str(target),
                        "operation": "edit",
                        "old": "missing string",
                        "new": "x",
                    }
                ],
                "issues": [],
            }
        )
        worker = _worker([resp], tmp_path)
        result = await worker.run()
        assert "Failed to edit" in result
        assert "file.txt" in result
