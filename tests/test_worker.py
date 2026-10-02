"""Tests for the WorkerAgent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel


def make_mock_client():
    """Create a mock LLMClient."""
    client = MagicMock()
    client.settings = Settings()
    client.complete = AsyncMock(return_value="Implemented feature successfully")
    return client


async def test_worker_run_returns_response():
    """Worker returns the LLM response."""
    client = make_mock_client()
    task = TaskModel(id="1", description="Add auth", files=["auth.py"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Implemented feature successfully"


async def test_worker_run_uses_correct_model():
    """Worker uses the worker_model from settings."""
    client = make_mock_client()
    task = TaskModel(id="1", description="Task", files=[])
    worker = WorkerAgent(task=task, client=client)
    await worker.run()
    call_args = client.complete.call_args
    assert call_args.kwargs.get("model") == client.settings.worker_model


async def test_worker_includes_task_description():
    """Worker prompt includes the task description."""
    client = make_mock_client()
    task = TaskModel(id="1", description="Implement JWT auth in src/auth.py", files=["src/auth.py"])
    worker = WorkerAgent(task=task, client=client)
    await worker.run()
    call_args = client.complete.call_args
    prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
    assert "Implement JWT auth" in prompt
    assert "src/auth.py" in prompt


async def test_worker_no_files():
    """Worker handles tasks with no file restrictions."""
    client = make_mock_client()
    task = TaskModel(id="1", description="Refactor", files=[])
    worker = WorkerAgent(task=task, client=client)
    await worker.run()
    call_args = client.complete.call_args
    prompt = call_args.args[0] if call_args.args else call_args.kwargs.get("prompt", "")
    assert "any" in prompt  # Should say 'any' when no files specified
