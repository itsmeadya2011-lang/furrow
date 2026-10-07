from __future__ import annotations

import json
from pathlib import Path

from furrow.config import Provider, Settings, TaskModel
from furrow.llm import LLMClient


class _DummySettings(Settings):
    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
        super().__init__(**kwargs)


class FakeClient(LLMClient):
    def __init__(self, workspace: Path, response_json: str | None = None, raw_response: str = ""):  # type: ignore[no-untyped-def]
        super().__init__(settings=_DummySettings(provider=Provider.ANTHROPIC, anthropic_api_key="x", workspace=workspace))
        self._response_json = response_json
        self._raw_response = raw_response
        self.written: dict[str, str] = {}
        self.complete_json_calls = 0
        self.complete_calls = 0

    def list_files(self, directory):  # type: ignore[override]
        return ["existing.py"]

    async def read_file(self, path):  # type: ignore[override]
        return "print('old')"

    async def write_file(self, path, content):  # type: ignore[override]
        self.written[str(path)] = content
        # Mirror real LLMClient behavior: create parent dirs and write to disk
        # so tests can assert on the actual workspace files.
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)

    async def complete_json(self, prompt, system="", model=None):  # type: ignore[override]
        self.complete_json_calls += 1
        if self._response_json is not None:
            return self._response_json
        raise ValueError("no json")

    async def complete(self, prompt, system="", model=None):  # type: ignore[override]
        self.complete_calls += 1
        return self._raw_response


def test_worker_applies_edits(tmp_path: Path):
    payload = json.dumps({
        "edits": [
            {"path": "src/foo.py", "action": "create", "content": "print('hi')"},
            {"path": "existing.py", "action": "modify", "content": "print('new')"},
        ],
        "summary": "created foo, modified existing",
    })
    client = FakeClient(tmp_path, response_json=payload)
    from furrow.agents.worker import WorkerAgent

    task = TaskModel(id="1", description="implement foo", files=["src/foo.py", "existing.py"])
    result = asyncio_run(WorkerAgent(task=task, client=client).run())

    assert result == "created foo, modified existing"
    assert (tmp_path / "src" / "foo.py").read_text() == "print('hi')"
    assert (tmp_path / "existing.py").read_text() == "print('new')"
    assert client.complete_json_calls == 1


def test_worker_falls_back_to_raw_text(tmp_path: Path):
    client = FakeClient(tmp_path, response_json=None, raw_response="just text")
    from furrow.agents.worker import WorkerAgent

    task = TaskModel(id="1", description="do thing", files=["a.py"])
    result = asyncio_run(WorkerAgent(task=task, client=client).run())

    assert result == "just text"
    assert client.complete_calls == 1
    assert client.written == {}


def test_worker_skips_malformed_edits(tmp_path: Path):
    payload = json.dumps({
        "edits": [
            {"path": "ok.py", "content": "x"},
            {},
            {"content": "no path"},
            {"path": "good.py", "content": "y"},
        ],
        "summary": "partial",
    })
    client = FakeClient(tmp_path, response_json=payload)
    from furrow.agents.worker import WorkerAgent

    task = TaskModel(id="1", description="mixed", files=[])
    result = asyncio_run(WorkerAgent(task=task, client=client).run())

    assert result == "partial"
    assert (tmp_path / "ok.py").read_text() == "x"
    assert (tmp_path / "good.py").read_text() == "y"
    assert "good.py" in str(client.written)


def asyncio_run(coro):
    import asyncio
    return asyncio.run(coro)