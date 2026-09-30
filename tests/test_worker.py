"""Tests for furrow.agents.worker."""
from __future__ import annotations

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel
from furrow.llm import LLMClient


class _FakeClient:
    def __init__(self, response: str = "Completed the task") -> None:
        self._response = response
        self.settings = LLMClient().settings
        self.complete_calls: list[tuple[str, str, str | None]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append((prompt, system, model))
        return self._response


@pytest.mark.asyncio
async def test_worker_returns_summary():
    task = TaskModel(id="1", description="Add a helper function")
    client = _FakeClient("Added helper.py with a utility function.")
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert result == "Added helper.py with a utility function."
    assert len(client.complete_calls) == 1


@pytest.mark.asyncio
async def test_worker_includes_task_in_prompt():
    task = TaskModel(
        id="2",
        description="Update README",
        files=["README.md"],
    )
    client = _FakeClient()
    worker = WorkerAgent(task=task, client=client)
    await worker.run()
    prompt = client.complete_calls[0][0]
    assert "Update README" in prompt
    assert "README.md" in prompt


@pytest.mark.asyncio
async def test_worker_no_files_lists_any():
    task = TaskModel(id="3", description="Investigate bug")
    client = _FakeClient()
    worker = WorkerAgent(task=task, client=client)
    await worker.run()
    prompt = client.complete_calls[0][0]
    assert "any" in prompt  # fallback when no files specified
