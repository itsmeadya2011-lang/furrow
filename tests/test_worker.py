from pathlib import Path
from types import SimpleNamespace

import pytest

from furrow.config import TaskModel
from furrow.agents.worker import WorkerAgent


class FakeLLMClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.settings = SimpleNamespace(worker_model="test-model")
        self.written = {}

    async def complete(self, prompt, system="", model=None):
        return self.responses.pop(0)

    async def read_file(self, path):
        return ""

    async def write_file(self, path, content):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.written[str(p)] = content

    def list_files(self, directory):
        return []


async def test_run_produces_summary():
    client = FakeLLMClient(["<<DONE>>All done.<<END>>"])
    task = TaskModel(id="1", description="do the thing")
    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()
    assert "All done" in summary


async def test_run_writes_file_on_write_block(tmp_path):
    target = tmp_path / "out.txt"
    client = FakeLLMClient(
        [
            f"<<WRITE>>{target}\nhello disk<<END>>",
            "<<DONE>>written.<<END>>",
        ]
    )
    task = TaskModel(id="1", description="write file")
    worker = WorkerAgent(task=task, client=client)
    summary = await worker.run()
    assert target.read_text() == "hello disk"
    assert "written" in summary
