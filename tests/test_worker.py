from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import TaskModel
from furrow.agents.worker import WorkerAgent


class MockLLMClient:
    def __init__(self):
        self.settings = MagicMock()
        self.settings.workspace = Path("/tmp/test_workspace")
        self.settings.worker_model = "test-model"
        self.read_file = AsyncMock(side_effect=self._read_side_effect)
        self.write_file = AsyncMock()
        self.list_files = MagicMock(return_value=["a.py", "b.py"])
        self.complete = AsyncMock(return_value="Done. No file changes.")
        self._files = {}

    def _read_side_effect(self, path):
        if str(path) in self._files:
            return self._files[str(path)]
        raise FileNotFoundError(f"not found: {path}")

    def set_file(self, path, content):
        self._files[str(path)] = content


def test_worker_reads_files():
    client = MockLLMClient()
    client.set_file("/tmp/test_workspace/auth.py", "print('old')")
    task = TaskModel(id="1", description="Update auth", files=["auth.py"])
    agent = WorkerAgent(task=task, client=client)
    import asyncio
    result = asyncio.run(agent.run())
    assert client.read_file.called
    assert "auth" in client.complete.call_args[0][0]


def test_worker_writes_file_blocks():
    client = MockLLMClient()
    client.complete = AsyncMock(return_value=(
        "---FILE: new.py---\nprint('hello')\n---END FILE---\nDone."
    ))
    task = TaskModel(id="1", description="Create file", files=[])
    agent = WorkerAgent(task=task, client=client)
    import asyncio
    result = asyncio.run(agent.run())
    assert client.write_file.called
    assert "new.py" in result


def test_worker_handles_missing_files():
    client = MockLLMClient()
    task = TaskModel(id="1", description="Task", files=["missing.py"])
    agent = WorkerAgent(task=task, client=client)
    import asyncio
    result = asyncio.run(agent.run())
    assert "Could not read" in result or "missing.py" in result


def test_worker_includes_workspace_files():
    client = MockLLMClient()
    task = TaskModel(id="1", description="Task", files=[])
    agent = WorkerAgent(task=task, client=client)
    import asyncio
    asyncio.run(agent.run())
    prompt = client.complete.call_args[0][0]
    assert "a.py" in prompt
    assert "b.py" in prompt