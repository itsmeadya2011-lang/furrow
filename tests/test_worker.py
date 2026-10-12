from __future__ import annotations

from pathlib import Path

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class _FakeLLMClient:
    """Captures write_file calls and returns a canned LLM response."""

    def __init__(self, response: str = "", settings: Settings | None = None) -> None:
        self.response = response
        self.written: list[tuple[str, str]] = []
        self.completed_prompts: list[str] = []
        self.settings = settings or Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.completed_prompts.append(prompt)
        return self.response

    async def write_file(self, path: str | Path, content: str) -> None:
        self.written.append((str(path), content))


@pytest.mark.asyncio
async def test_worker_writes_file_blocks(tmp_path: Path) -> None:
    response = (
        "FILE: hello.py\n"
        "```python\n"
        "print('hi')\n"
        "```\n"
        "SUMMARY: created hello.py"
    )
    client = _FakeLLMClient(response=response)
    task = TaskModel(id="1", description="create hello.py", files=["hello.py"])
    worker = WorkerAgent(task=task, client=client)  # type: ignore[arg-type]
    summary = await worker.run()

    assert client.written == [("hello.py", "print('hi')\n")]
    assert summary == "created hello.py"


@pytest.mark.asyncio
async def test_worker_no_file_blocks_returns_summary_only() -> None:
    response = "SUMMARY: nothing to do"
    client = _FakeLLMClient(response=response)
    task = TaskModel(id="1", description="noop")
    worker = WorkerAgent(task=task, client=client)  # type: ignore[arg-type]
    summary = await worker.run()

    assert client.written == []
    assert summary == "nothing to do"


@pytest.mark.asyncio
async def test_worker_strips_trailing_carriage_return(tmp_path: Path) -> None:
    response = (
        "FILE: a.py\n"
        "```python\n"
        "x = 1\r\n"
        "```\n"
    )
    client = _FakeLLMClient(response=response)
    task = TaskModel(id="1", description="edit a.py", files=["a.py"])
    worker = WorkerAgent(task=task, client=client)  # type: ignore[arg-type]
    await worker.run()

    assert client.written == [("a.py", "x = 1\n")]


@pytest.mark.asyncio
async def test_worker_multiple_file_blocks(tmp_path: Path) -> None:
    response = (
        "FILE: a.py\n"
        "```python\n"
        "a = 1\n"
        "```\n"
        "FILE: b.py\n"
        "```python\n"
        "b = 2\n"
        "```\n"
        "SUMMARY: wrote two files"
    )
    client = _FakeLLMClient(response=response)
    task = TaskModel(id="1", description="two files", files=["a.py", "b.py"])
    worker = WorkerAgent(task=task, client=client)  # type: ignore[arg-type]
    await worker.run()

    assert ("a.py", "a = 1\n") in client.written
    assert ("b.py", "b = 2\n") in client.written