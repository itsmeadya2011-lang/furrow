from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class FakeClient:
    def __init__(self, response: str, workspace: Path) -> None:
        self._response = response
        self.settings = SimpleNamespace(worker_model="x", workspace=Path(workspace))
        self.written: dict[str, str] = {}

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return self._response

    async def read_file(self, path: str | Path) -> str:
        key = str(path)
        if key in self.written:
            return self.written[key]
        raise FileNotFoundError(key)

    async def write_file(self, path: str | Path, content: str) -> None:
        self.written[str(path)] = content


def _payload(files: list[dict], summary: str = "done") -> str:
    return json.dumps({"files": files, "summary": summary})


async def test_worker_writes_files(tmp_path: Path) -> None:
    response = _payload(
        [
            {"path": "a.py", "content": "print('a')"},
            {"path": "b.py", "content": "print('b')"},
        ],
        summary="wrote two files",
    )
    client = FakeClient(response, workspace=tmp_path)
    task = TaskModel(id="1", description="make files", files=["a.py", "b.py"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert result == "wrote two files"
    expected = {
        str((tmp_path / "a.py").resolve()): "print('a')",
        str((tmp_path / "b.py").resolve()): "print('b')",
    }
    assert client.written == expected


async def test_worker_handles_fenced_json(tmp_path: Path) -> None:
    payload = _payload(
        [{"path": "main.py", "content": "print('hello')"}],
        summary="created main entry point",
    )
    response = f"```json\n{payload}\n```"
    client = FakeClient(response, workspace=tmp_path)
    task = TaskModel(id="1", description="add entry point", files=["main.py"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert result == "created main entry point"
    assert client.written[str((tmp_path / "main.py").resolve())] == "print('hello')"


async def test_worker_parse_failure_returns_warning(tmp_path: Path) -> None:
    response = "Here is what I did: nothing useful."
    client = FakeClient(response, workspace=tmp_path)
    task = TaskModel(id="1", description="do a thing", files=["a.py"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert result.startswith("WARNING: could not parse worker JSON output; raw response below:\n")
    assert response in result
    assert client.written == {}


async def test_worker_no_files(tmp_path: Path) -> None:
    response = _payload([], summary="nothing to do")
    client = FakeClient(response, workspace=tmp_path)
    task = TaskModel(id="1", description="noop", files=[])
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert result == "nothing to do"
    assert client.written == {}


async def test_worker_safely_normalizes_absolute_path(tmp_path: Path) -> None:
    response = _payload(
        [{"path": "/etc/passwd", "content": "hacked"}],
        summary="ok",
    )
    client = FakeClient(response, workspace=tmp_path)
    task = TaskModel(id="1", description="safe write", files=["/etc/passwd"])
    worker = WorkerAgent(task=task, client=client)

    result = await worker.run()

    assert result == "ok"
    assert client.written == {str((tmp_path / "passwd").resolve()): "hacked"}
