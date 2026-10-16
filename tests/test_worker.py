from unittest.mock import AsyncMock

from furrow.config import Settings, TaskModel
from furrow.agents.worker import WorkerAgent
from furrow.llm import LLMClient


async def test_worker_returns_summary_string():
    client = LLMClient(settings=Settings())
    client.complete = AsyncMock(return_value="implemented the feature successfully")
    task = TaskModel(id="1", description="implement feature", files=["a.py"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()
    assert isinstance(result, str)
    assert result == "implemented the feature successfully"


async def test_worker_uses_correct_model():
    client = LLMClient(settings=Settings())
    client.complete = AsyncMock(return_value="done")
    task = TaskModel(id="1", description="implement feature")
    worker = WorkerAgent(task=task, client=client)
    await worker.run()
    _, kwargs = client.complete.call_args
    assert kwargs["model"] == client.settings.worker_model
