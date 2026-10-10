"""Tests for the WorkerAgent in furrow.agents.worker."""

import asyncio
from pathlib import Path

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


class FakeWorkerClient:
    """Returns a fixed LLM response containing a fenced code block."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.complete_calls = 0
        self.settings = Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls += 1
        return self.response


def test_worker_writes_fenced_code_block_to_workspace(tmp_path: Path) -> None:
    response = (
        "Here is the implementation:\n\n"
        "```python\nsrc/app.py\nprint('hello from furrow')\n```\n"
    )
    client = FakeWorkerClient(response)
    task = TaskModel(id="1", description="write a script", files=["src/app.py"])

    agent = WorkerAgent(task=task, client=client)
    # Override the workspace the agent reads from its client settings.
    client.settings.workspace = tmp_path
    agent.workspace = tmp_path

    summary = asyncio.run(agent.run())

    written = tmp_path / "src" / "app.py"
    assert written.exists()
    assert "hello from furrow" in written.read_text()
    assert "src/app.py" in summary


def test_worker_returns_summary_mentioning_written_files(tmp_path: Path) -> None:
    response = (
        "```python\nlib/mod.py\nx = 1\n```\n\n"
        "```python\nlib/util.py\ny = 2\n```\n"
    )
    client = FakeWorkerClient(response)
    task = TaskModel(id="2", description="write modules", files=["lib/mod.py", "lib/util.py"])

    agent = WorkerAgent(task=task, client=client)
    client.settings.workspace = tmp_path
    agent.workspace = tmp_path

    summary = asyncio.run(agent.run())

    assert "lib/mod.py" in summary
    assert "lib/util.py" in summary
    assert (tmp_path / "lib" / "mod.py").exists()
    assert (tmp_path / "lib" / "util.py").exists()


def test_worker_with_no_code_blocks_returns_response(tmp_path: Path) -> None:
    response = "I couldn't produce any files for this task."
    client = FakeWorkerClient(response)
    task = TaskModel(id="3", description="impossible task", files=[])

    agent = WorkerAgent(task=task, client=client)
    client.settings.workspace = tmp_path
    agent.workspace = tmp_path

    summary = asyncio.run(agent.run())

    assert summary == response
    # Nothing should have been written.
    assert list(tmp_path.rglob("*")) == []