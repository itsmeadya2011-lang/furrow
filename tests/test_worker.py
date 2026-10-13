"""Tests for the WorkerAgent."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.prompts import WORKER_PROMPT
from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


@pytest.mark.asyncio
async def test_worker_returns_llm_response():
    """Worker should return the LLM's completion response."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "worker-model"
    client.complete = AsyncMock(return_value="Implemented auth module")

    task = TaskModel(id="1", description="Add JWT auth", files=["src/auth.py"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "Implemented auth module"
    client.complete.assert_called_once()


@pytest.mark.asyncio
async def test_worker_passes_task_description():
    """Worker should include the task description in the prompt."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "worker-model"
    client.complete = AsyncMock(return_value="done")

    task = TaskModel(id="1", description="Fix the login bug", files=[])
    worker = WorkerAgent(task=task, client=client)
    await worker.run()

    call_args = client.complete.call_args
    prompt = call_args[0][0]
    assert "Fix the login bug" in prompt


@pytest.mark.asyncio
async def test_worker_passes_files():
    """Worker should include file paths in the prompt."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "worker-model"
    client.complete = AsyncMock(return_value="done")

    task = TaskModel(id="1", description="task", files=["src/foo.py", "tests/test_foo.py"])
    worker = WorkerAgent(task=task, client=client)
    await worker.run()

    call_args = client.complete.call_args
    prompt = call_args[0][0]
    assert "src/foo.py" in prompt
    assert "tests/test_foo.py" in prompt


@pytest.mark.asyncio
async def test_worker_no_files_uses_any():
    """Worker should use 'any' when no files are specified."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "worker-model"
    client.complete = AsyncMock(return_value="done")

    task = TaskModel(id="1", description="task", files=[])
    worker = WorkerAgent(task=task, client=client)
    await worker.run()

    call_args = client.complete.call_args
    prompt = call_args[0][0]
    assert "any" in prompt


@pytest.mark.asyncio
async def test_worker_uses_worker_model():
    """Worker should use the worker_model from settings."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "claude-3-5-sonnet"
    client.complete = AsyncMock(return_value="done")

    task = TaskModel(id="1", description="task")
    worker = WorkerAgent(task=task, client=client)
    await worker.run()

    call_kwargs = client.complete.call_args
    # model should be passed as keyword arg
    assert "claude-3-5-sonnet" in str(call_kwargs)
