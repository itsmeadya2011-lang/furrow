"""Tests for the WorkerAgent."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


@pytest.mark.asyncio
async def test_worker_returns_result(mock_llm_client: MagicMock):
    """Worker returns the LLM response string."""
    mock_llm_client.complete = AsyncMock(return_value="Implemented feature X.")

    task = TaskModel(id="1", description="Implement feature X", files=["src/feature.py"])
    worker = WorkerAgent(task=task, client=mock_llm_client)
    result = await worker.run()

    assert result == "Implemented feature X."
    mock_llm_client.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_prompt_includes_task_description(mock_llm_client: MagicMock):
    """The prompt sent to LLM includes the task description."""
    mock_llm_client.complete = AsyncMock(return_value="ok")

    task = TaskModel(id="1", description="Refactor module Y", files=["src/y.py"])
    worker = WorkerAgent(task=task, client=mock_llm_client)
    await worker.run()

    call_args = mock_llm_client.complete.call_args
    prompt = call_args.args[0]
    assert "Refactor module Y" in prompt
    assert "src/y.py" in prompt


@pytest.mark.asyncio
async def test_worker_prompt_handles_no_files(mock_llm_client: MagicMock):
    """When task has no files, prompt uses 'any'."""
    mock_llm_client.complete = AsyncMock(return_value="ok")

    task = TaskModel(id="1", description="Review code")
    worker = WorkerAgent(task=task, client=mock_llm_client)
    await worker.run()

    call_args = mock_llm_client.complete.call_args
    prompt = call_args.args[0]
    assert "any" in prompt


@pytest.mark.asyncio
async def test_worker_uses_worker_model(mock_llm_client: MagicMock):
    """Worker uses the worker_model from settings."""
    mock_llm_client.complete = AsyncMock(return_value="ok")

    task = TaskModel(id="1", description="Task", files=["a.py"])
    worker = WorkerAgent(task=task, client=mock_llm_client)
    await worker.run()

    call_kwargs = mock_llm_client.complete.call_args.kwargs
    assert call_kwargs["model"] == mock_llm_client.settings.worker_model
