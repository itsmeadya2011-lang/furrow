import pytest
from unittest.mock import AsyncMock

from furrow.agents.worker import WorkerAgent
from furrow.config import TaskModel


async def test_worker_run_calls_llm_and_returns_text():
    stub_client = AsyncMock()
    stub_client.complete = AsyncMock(return_value="worker-result")
    task = TaskModel(id="1", description="do work")
    worker = WorkerAgent(task=task, client=stub_client)
    result = await worker.run()
    assert result == "worker-result"
    stub_client.complete.assert_called_once()
