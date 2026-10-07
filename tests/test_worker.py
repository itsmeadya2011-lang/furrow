from __future__ import annotations

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


class MockWorkerClient:
    """Mock LLM client for worker tests."""

    def __init__(self):
        self.settings = type("S", (), {"worker_model": "mock"})()
        self.responses = []
        self.writes = []

    def queue(self, response: str):
        self.responses.append(response)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        if self.responses:
            return self.responses.pop(0)
        return ""

    async def write_file(self, path: str, content: str) -> None:
        self.writes.append((path, content))


def test_worker_writes_file_from_block():
    """Worker parses file blocks and writes files."""
    client = MockWorkerClient()
    client.queue('```file:hello.py\nprint("hi")\n```\nSUMMARY: created hello.py')
    task = TaskModel(id="1", description="create hello.py", files=["hello.py"])
    agent = WorkerAgent(task=task, client=client)
    import asyncio
    summary = asyncio.run(agent.run())
    assert "created hello.py" in summary
    assert len(client.writes) == 1
    assert client.writes[0][0] == "hello.py"
    assert 'print("hi")' in client.writes[0][1]


def test_worker_writes_multiple_files():
    """Worker handles multiple file blocks."""
    client = MockWorkerClient()
    client.queue(
        '```file:a.py\nx = 1\n```\n```file:b.py\ny = 2\n```\nSUMMARY: two files'
    )
    task = TaskModel(id="1", description="create a.py and b.py", files=["a.py", "b.py"])
    agent = WorkerAgent(task=task, client=client)
    import asyncio
    summary = asyncio.run(agent.run())
    assert "two files" in summary
    assert len(client.writes) == 2
    paths = [p for p, _ in client.writes]
    assert "a.py" in paths
    assert "b.py" in paths


def test_worker_no_files_returns_default():
    """Worker returns default summary when no file blocks found."""
    client = MockWorkerClient()
    client.queue("I don't have any file changes to make.")
    task = TaskModel(id="1", description="do nothing", files=[])
    agent = WorkerAgent(task=task, client=client)
    import asyncio
    summary = asyncio.run(agent.run())
    assert "no file changes" in summary.lower()
    assert len(client.writes) == 0


def test_worker_strips_path():
    """Worker strips whitespace from file paths."""
    client = MockWorkerClient()
    client.queue('```file:  spaced.py  \nval = 42\n```\nSUMMARY: done')
    task = TaskModel(id="1", description="create spaced.py", files=["spaced.py"])
    agent = WorkerAgent(task=task, client=client)
    import asyncio
    asyncio.run(agent.run())
    assert client.writes[0][0] == "spaced.py"
