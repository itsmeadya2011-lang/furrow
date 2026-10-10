import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class FakeLLMClient:
    def __init__(self, response: str, workspace):
        self.settings = SimpleNamespace(workspace=Path(workspace), worker_model="fake-model")
        self.response = response
        self.written = {}

    async def complete(self, prompt, system="", model=None):
        return self.response

    async def write_file(self, path, content):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.written[str(path)] = content


@pytest.mark.asyncio
async def test_worker_writes_files_from_valid_json(tmp_path):
    payload = {
        "summary": "implemented a.py",
        "edits": [{"path": "a.py", "content": "print('hello')\n"}],
    }
    fake = FakeLLMClient(json.dumps(payload), tmp_path)
    agent = WorkerAgent(task=TaskModel(id="1", description="do thing"), client=fake)

    summary = await agent.run()

    assert (tmp_path / "a.py").read_text() == "print('hello')\n"
    assert "1 files written" in summary
    assert "implemented a.py" in summary


@pytest.mark.asyncio
async def test_worker_rejects_path_escapes(tmp_path):
    payload = {
        "summary": "nope",
        "edits": [
            {"path": "../evil.py", "content": "x"},
            {"path": "/tmp/evil.py", "content": "x"},
        ],
    }
    fake = FakeLLMClient(json.dumps(payload), tmp_path)
    agent = WorkerAgent(task=TaskModel(id="1", description="do thing"), client=fake)

    summary = await agent.run()

    assert "rejected" in summary
    assert "../evil.py" in summary
    assert "/tmp/evil.py" in summary
    assert fake.written == {}
    assert not (tmp_path.parent / "evil.py").exists()
    assert not Path("/tmp/evil.py").exists()


@pytest.mark.asyncio
async def test_worker_returns_raw_response_when_not_json(tmp_path):
    fake = FakeLLMClient("I couldn't do it", tmp_path)
    agent = WorkerAgent(task=TaskModel(id="1", description="do thing"), client=fake)

    summary = await agent.run()

    assert summary == "I couldn't do it"


@pytest.mark.asyncio
async def test_worker_empty_edits_returns_summary(tmp_path):
    payload = {"summary": "nothing to do", "edits": []}
    fake = FakeLLMClient(json.dumps(payload), tmp_path)
    agent = WorkerAgent(task=TaskModel(id="1", description="do thing"), client=fake)

    summary = await agent.run()

    assert summary == "nothing to do"
    assert fake.written == {}