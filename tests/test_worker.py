import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.agents.worker import WorkerAgent
from furrow.config import Settings, TaskModel
from furrow.llm import LLMClient


def make_client():
    client = MagicMock(spec=LLMClient)
    client.settings = Settings()
    return client


@pytest.mark.asyncio
async def test_worker_parses_json_and_writes_files():
    client = make_client()
    client.complete = AsyncMock(return_value=json.dumps({
        "files": {"hello.py": "print('hi')"},
        "summary": "wrote hello.py",
    }))
    client.write_file = AsyncMock()

    task = TaskModel(id="1", description="write hello.py", files=["hello.py"])
    agent = WorkerAgent(task=task, client=client)

    result = await agent.run()
    assert result == "wrote hello.py"
    client.write_file.assert_called_once_with("hello.py", "print('hi')")


@pytest.mark.asyncio
async def test_worker_fallback_to_raw_text_when_json_parsing_fails():
    client = make_client()
    client.complete = AsyncMock(return_value="I cannot parse this response.")
    client.write_file = AsyncMock()

    task = TaskModel(id="1", description="do something", files=[])
    agent = WorkerAgent(task=task, client=client)

    result = await agent.run()
    assert result == "I cannot parse this response."
    client.write_file.assert_not_called()


@pytest.mark.asyncio
async def test_worker_uses_default_summary_when_no_summary_provided():
    client = make_client()
    client.complete = AsyncMock(return_value=json.dumps({
        "files": {"a.txt": "content"},
    }))
    client.write_file = AsyncMock()

    task = TaskModel(id="1", description="write a.txt", files=["a.txt"])
    agent = WorkerAgent(task=task, client=client)

    result = await agent.run()
    assert result == "Wrote 1 file(s): a.txt"
