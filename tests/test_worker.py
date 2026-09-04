import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.planner import _strip_fences
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel, Settings


def test_worker_strip_fences():
    assert _strip_fences("```json\n{'a': 1}\n```") == "{'a': 1}"
    assert _strip_fences("no fences here") == "no fences here"
    assert _strip_fences("```\nplain\n```") == "plain"


@pytest.mark.asyncio
async def test_worker_writes_files_success(tmp_path, monkeypatch):
    response = json.dumps({
        "files_written": [{"path": "hello.txt", "content": "Hello, world!"}],
        "summary": "Wrote hello.txt",
    })
    client = MagicMock()
    client.complete = AsyncMock(return_value=response)
    client.settings = Settings(workspace=tmp_path)

    monkeypatch.chdir(tmp_path)
    agent = WorkerAgent(task=TaskModel(id="1", description="write hello"), client=client)
    result = await agent.run()
    assert result == "Wrote hello.txt"
    assert (tmp_path / "hello.txt").read_text() == "Hello, world!"


@pytest.mark.asyncio
async def test_worker_falls_back_on_malformed():
    client = MagicMock()
    client.complete = AsyncMock(return_value="this is not json")
    client.settings = Settings()

    agent = WorkerAgent(task=TaskModel(id="1", description="do stuff"), client=client)
    result = await agent.run()
    assert result == "this is not json"
