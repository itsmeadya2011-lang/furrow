from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


def test_worker_writes_files_from_response() -> None:
    client = MagicMock()
    client.settings.worker_model = "test-model"
    client.complete = AsyncMock(return_value=(
        "FILE: hello.py\nprint('hello')\n\n"
        "FILE: utils.py\ndef foo():\n    pass\n"
    ))
    client.write_file = AsyncMock()

    task = TaskModel(id="1", description="create files", files=["hello.py", "utils.py"])
    agent = WorkerAgent(task=task, client=client)
    asyncio.run(agent.run())

    assert client.write_file.call_count == 2
    paths = {call.args[0] for call in client.write_file.call_args_list}
    assert Path("hello.py") in paths
    assert Path("utils.py") in paths


def test_worker_no_files_in_response() -> None:
    client = MagicMock()
    client.settings.worker_model = "test-model"
    client.complete = AsyncMock(return_value="Just a summary, no files.")
    client.write_file = AsyncMock()

    task = TaskModel(id="1", description="do thing", files=[])
    agent = WorkerAgent(task=task, client=client)
    result = asyncio.run(agent.run())

    assert client.write_file.call_count == 0
    assert "summary" in result.lower()


def test_worker_handles_malformed_file_blocks() -> None:
    client = MagicMock()
    client.settings.worker_model = "test-model"
    client.complete = AsyncMock(return_value="FILE: broken\nno closing")
    client.write_file = AsyncMock()

    task = TaskModel(id="1", description="broken", files=[])
    agent = WorkerAgent(task=task, client=client)
    asyncio.run(agent.run())
    # Should not raise
    assert True