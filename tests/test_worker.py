"""Tests for furrow.agents.worker: WorkerAgent.run and _parse_response."""

import json
from unittest.mock import AsyncMock

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class MockLLMClient:
    def __init__(self, settings: Settings, response: str) -> None:
        self.settings = settings
        self.complete = AsyncMock(return_value=response)
        self.read_file = AsyncMock()
        self.write_file = AsyncMock()


def make_settings() -> Settings:
    return Settings(provider="anthropic", anthropic_api_key="sk-test")


def make_client(response: str) -> "MockLLMClient":
    return MockLLMClient(settings=make_settings(), response=response)


def test_parse_response_valid_json():
    worker = WorkerAgent(task=TaskModel(id="1", description="t"))
    response = json.dumps({
        "changes": [{"file": "a.py", "content": "new content"}],
        "summary": "updated a.py",
    })
    changes, summary = worker._parse_response(response)
    assert changes == [{"file": "a.py", "content": "new content"}]
    assert summary == "updated a.py"


def test_parse_response_markdown_wrapped_json():
    worker = WorkerAgent(task=TaskModel(id="1", description="t"))
    response = '```json\n{"changes": [{"file": "b.py", "content": "x"}], "summary": "done"}\n```'
    changes, summary = worker._parse_response(response)
    assert changes == [{"file": "b.py", "content": "x"}]
    assert summary == "done"


def test_parse_response_fallback_on_non_json():
    worker = WorkerAgent(task=TaskModel(id="1", description="t"))
    response = "here is some free text without json"
    changes, summary = worker._parse_response(response)
    assert changes == []
    assert summary == response


def test_parse_response_with_missing_summary_field():
    worker = WorkerAgent(task=TaskModel(id="1", description="t"))
    response = json.dumps({"changes": [{"file": "a.py", "content": "x"}]})
    changes, summary = worker._parse_response(response)
    assert changes == []
    assert summary == response


def test_parse_response_with_missing_changes_field():
    worker = WorkerAgent(task=TaskModel(id="1", description="t"))
    response = json.dumps({"summary": "only summary"})
    changes, summary = worker._parse_response(response)
    assert changes == []
    assert summary == response


def test_parse_response_empty_changes():
    worker = WorkerAgent(task=TaskModel(id="1", description="t"))
    response = json.dumps({"changes": [], "summary": "no changes needed"})
    changes, summary = worker._parse_response(response)
    assert changes == []
    assert summary == "no changes needed"


async def test_run_reads_task_files_sends_prompt_writes_changes(tmp_path):
    client = make_client(json.dumps({
        "changes": [{"file": str(tmp_path / "out.py"), "content": "written"}],
        "summary": "wrote file",
    }))
    task = TaskModel(id="1", description="write file", files=["src/main.py"])
    client.read_file.return_value = "original content"
    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()
    assert summary == "wrote file"
    client.read_file.assert_awaited_once_with("src/main.py")
    client.complete.assert_awaited_once()
    client.write_file.assert_awaited_once_with(str(tmp_path / "out.py"), "written")


async def test_run_handles_missing_files(tmp_path):
    client = make_client(json.dumps({"changes": [], "summary": "ok"}))
    task = TaskModel(id="1", description="no file", files=["missing.py"])
    client.read_file.side_effect = FileNotFoundError("missing")
    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()
    assert summary == "ok"
    client.write_file.assert_not_awaited()


async def test_run_with_no_task_files():
    client = make_client(json.dumps({"changes": [], "summary": "nothing to do"}))
    task = TaskModel(id="1", description="free work")
    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()
    assert summary == "nothing to do"
    client.read_file.assert_not_awaited()
