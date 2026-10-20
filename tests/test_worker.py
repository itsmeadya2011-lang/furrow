import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class FakeClient:
    """Deterministic fake LLMClient: no network, no real LLM, records writes."""

    def __init__(self, workspace: Path, initial_files: dict[str, str] | None = None) -> None:
        self.settings = SimpleNamespace(workspace=workspace, worker_model="test-model")
        self.initial_files = dict(initial_files or {})
        self.complete_response = ""
        self.complete_calls: list[tuple[str, str | None]] = []
        self.writes: list[tuple[str, str]] = []
        self.read_errors: set[str] = set()

    async def complete(self, prompt: str, model: str | None = None, system: str = "") -> str:
        self.complete_calls.append((prompt, model))
        return self.complete_response

    async def read_file(self, path: str | Path) -> str:
        if str(path) in self.read_errors:
            raise FileNotFoundError(f"no such file: {path}")
        return self.initial_files.get(str(path), "")

    async def write_file(self, path: str | Path, content: str) -> None:
        self.writes.append((str(path), content))

    def list_files(self, directory: str | Path) -> list[str]:
        return sorted(self.initial_files)


def _task(files: list[str] | None = None, description: str = "implement the thing") -> TaskModel:
    return TaskModel(id="1", description=description, files=files or [])


async def test_worker_writes_files_from_the_json_response(tmp_path: Path) -> None:
    client = FakeClient(tmp_path, {"src/app.py": "print('old')\n"})
    client.complete_response = json.dumps(
        {"summary": "implemented hello", "files": {"src/app.py": "print('hello')\n"}}
    )
    worker = WorkerAgent(task=_task(["src/app.py"], "implement hello"), client=client)
    result = await worker.run()

    assert client.writes == [(str(tmp_path / "src/app.py"), "print('hello')\n")]
    assert "implemented hello" in result
    assert "wrote:" in result


async def test_worker_prompt_contains_the_task_and_current_file_contents(tmp_path: Path) -> None:
    client = FakeClient(tmp_path, {"src/app.py": "print('old')\n"})
    client.complete_response = json.dumps({"summary": "s", "files": {}})
    worker = WorkerAgent(task=_task(["src/app.py"], "implement hello"), client=client)
    await worker.run()

    prompt, model = client.complete_calls[0]
    assert "implement hello" in prompt
    assert "print('old')" in prompt
    assert model == "test-model"


async def test_worker_returns_a_useful_summary(tmp_path: Path) -> None:
    client = FakeClient(tmp_path)
    client.complete_response = json.dumps(
        {"summary": "did stuff", "files": {"a.py": "A\n", "b.py": "B\n"}}
    )
    worker = WorkerAgent(task=_task(["a.py", "b.py"]), client=client)
    result = await worker.run()

    assert "did stuff" in result
    assert str(tmp_path / "a.py") in result
    assert str(tmp_path / "b.py") in result
    assert "wrote:" in result


async def test_worker_ignores_invalid_json(tmp_path: Path) -> None:
    client = FakeClient(tmp_path)
    client.complete_response = "Sorry, I cannot do that in JSON."
    worker = WorkerAgent(task=_task(["a.py"]), client=client)
    result = await worker.run()

    assert client.writes == []
    assert "No machine-readable patch" in result
    assert "Sorry, I cannot do that in JSON." in result


async def test_worker_strips_markdown_code_fences(tmp_path: Path) -> None:
    client = FakeClient(tmp_path)
    payload = json.dumps({"summary": "fenced", "files": {"a.py": "x = 1\n"}})
    client.complete_response = f"```json\n{payload}\n```"
    worker = WorkerAgent(task=_task(["a.py"]), client=client)
    result = await worker.run()

    assert client.writes == [(str(tmp_path / "a.py"), "x = 1\n")]
    assert "fenced" in result


async def test_worker_rejects_paths_escaping_the_workspace(tmp_path: Path) -> None:
    client = FakeClient(tmp_path)
    client.complete_response = json.dumps(
        {
            "summary": "tried to escape",
            "files": {"../evil.py": "bad\n", "src/ok.py": "ok\n"},
        }
    )
    worker = WorkerAgent(task=_task([]), client=client)
    result = await worker.run()

    written_paths = [path for path, _ in client.writes]
    assert not any("evil.py" in path for path in written_paths)
    assert any("ok.py" in path for path in written_paths)
    assert "rejected" in result


async def test_worker_survives_a_missing_input_file(tmp_path: Path) -> None:
    client = FakeClient(tmp_path, {"keep.py": "keep\n"})
    client.read_errors = {"missing.py"}
    client.complete_response = json.dumps({"summary": "done", "files": {"keep.py": "keep2\n"}})
    worker = WorkerAgent(task=_task(["missing.py", "keep.py"]), client=client)
    result = await worker.run()  # must not raise

    assert client.writes == [(str(tmp_path / "keep.py"), "keep2\n")]
    assert "done" in result


async def test_worker_raises_when_writing_fails(tmp_path: Path) -> None:
    class FailingClient(FakeClient):
        async def write_file(self, path: str | Path, content: str) -> None:
            self.writes.append((str(path), content))
            raise OSError("disk full")

    client = FailingClient(tmp_path)
    client.complete_response = json.dumps({"summary": "s", "files": {"a.py": "A\n", "b.py": "B\n"}})
    worker = WorkerAgent(task=_task(["a.py", "b.py"]), client=client)

    with pytest.raises(OSError):
        await worker.run()

    # Remaining files were still written first (partial patch applied).
    assert len(client.writes) == 2
