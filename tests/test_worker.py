import asyncio
from pathlib import Path

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class FakeClient:
    class Settings:
        worker_model = "test-model"

    def __init__(self) -> None:
        self.settings = self.Settings()
        self.written: dict[str, str] = {}
        self.completed_text = ""

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return self.completed_text

    async def write_file(self, path: str, content: str) -> None:
        self.written[str(Path(path))] = content


def test_worker_writes_files(tmp_path: Path) -> None:
    client = FakeClient()
    target = tmp_path / "hello.txt"
    client.completed_text = (
        '{"changes": [{"file": "' + str(target) + '", "content": "hi there"}], '
        '"summary": "wrote hello"}'
    )
    task = TaskModel(id="1", description="write hello", files=[str(target)])
    worker = WorkerAgent(task=task, client=client)
    result = asyncio.run(worker.run())

    assert client.written == {str(target): "hi there"}
    assert "Wrote files:" in result
    assert str(target) in result


def test_worker_falls_back_to_raw_text_on_invalid_json() -> None:
    client = FakeClient()
    client.completed_text = "just a plain summary, no json"
    task = TaskModel(id="1", description="do thing", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = asyncio.run(worker.run())

    assert result == "just a plain summary, no json"
    assert client.written == {}
