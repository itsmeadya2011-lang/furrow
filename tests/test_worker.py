import json
from pathlib import Path

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class FakeClient:
    def __init__(self, settings: Settings, responses: list[str]) -> None:
        self.settings = settings
        self.responses = iter(responses)
        self.prompts: list[str] = []
        self.read_paths: list[Path] = []
        self.write_calls: list[tuple[Path, str]] = []
        self.list_directories: list[Path] = []

    async def complete(self, prompt: str, model: str | None = None) -> str:
        self.prompts.append(prompt)
        return next(self.responses)

    async def read_file(self, path: Path) -> str:
        self.read_paths.append(path)
        return Path(path).read_text()

    async def write_file(self, path: Path, content: str) -> None:
        self.write_calls.append((path, content))
        Path(path).write_text(content)

    def list_files(self, directory: Path) -> list[str]:
        self.list_directories.append(directory)
        return sorted(
            str(path.relative_to(directory))
            for path in Path(directory).rglob("*")
            if path.is_file()
        )


def task(description: str = "update the worker") -> TaskModel:
    return TaskModel(id="1", description=description)


@pytest.mark.asyncio
async def test_worker_read_write_list_and_done_flow(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("original")
    settings = Settings(workspace=tmp_path)
    client = FakeClient(
        settings,
        [
            json.dumps({"action": "read_file", "path": "source.txt"}),
            json.dumps(
                {
                    "action": "write_file",
                    "path": "output.txt",
                    "content": "created",
                }
            ),
            json.dumps({"action": "list_files", "directory": "."}),
            json.dumps({"action": "done", "summary": "update complete"}),
        ],
    )

    result = await WorkerAgent(task=task(), client=client).run()

    assert result == "update complete"
    assert client.read_paths == [source.resolve()]
    assert client.write_calls == [(tmp_path / "output.txt", "created")]
    assert client.list_directories == [tmp_path.resolve()]
    assert (tmp_path / "output.txt").read_text() == "created"
    assert len(client.prompts) == 4
    assert "tool: " in client.prompts[1]
    assert '"action": "read_file"' in client.prompts[1]
    assert '"status": "success"' in client.prompts[1]
    assert '"ok": true' in client.prompts[1]


@pytest.mark.asyncio
async def test_worker_stops_on_malformed_final_response(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path)
    client = FakeClient(
        settings,
        [
            json.dumps(
                {
                    "action": "write_file",
                    "path": "result.txt",
                    "content": "partial",
                }
            ),
            "final prose from the model",
        ],
    )

    result = await WorkerAgent(task=task(), client=client).run()

    assert result == "final prose from the model"
    assert len(client.prompts) == 2
    assert (tmp_path / "result.txt").read_text() == "partial"


@pytest.mark.asyncio
async def test_worker_rejects_path_traversal(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path)
    client = FakeClient(
        settings,
        [
            json.dumps({"action": "read_file", "path": "../secret.txt"}),
            json.dumps({"action": "done", "summary": "safe completion"}),
        ],
    )

    result = await WorkerAgent(task=task(), client=client).run()

    assert result == "safe completion"
    assert client.read_paths == []
    assert "outside the workspace" in client.prompts[1]


@pytest.mark.asyncio
async def test_worker_stops_at_five_model_turns(tmp_path: Path) -> None:
    settings = Settings(workspace=tmp_path)
    client = FakeClient(
        settings,
        [json.dumps({"action": "list_files", "directory": "."}) for _ in range(5)],
    )

    result = await WorkerAgent(task=task(), client=client).run()

    assert result == "Stopped after 5 model turns without a done action."
    assert len(client.prompts) == 5
