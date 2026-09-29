"""Tests for WorkerAgent file writing and LLMClient provider dispatch."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.worker import _FILE_BLOCK_RE, WorkerAgent
from furrow.config import Provider, Settings, TaskModel
from furrow.llm import LLMClient

# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------

@pytest.fixture
def worker_task() -> TaskModel:
    return TaskModel(id="1", description="write a file")


@pytest.fixture
def worker_client() -> MagicMock:
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.worker_model = "test-model"
    client.write_file = AsyncMock()
    return client


@pytest.mark.asyncio
async def test_worker_writes_single_file(worker_task: TaskModel, worker_client: MagicMock) -> None:
    file_content = "print('hello')\n"
    response = f"FILE: /tmp/test_worker_output.py\n```python\n{file_content}```"
    worker_client.complete = AsyncMock(return_value=response)

    worker = WorkerAgent(task=worker_task, client=worker_client)
    result = await worker.run()

    assert "/tmp/test_worker_output.py" in result
    worker_client.write_file.assert_awaited_once_with("/tmp/test_worker_output.py", file_content)


@pytest.mark.asyncio
async def test_worker_no_file_blocks_returns_response(worker_task: TaskModel, worker_client: MagicMock) -> None:
    response = "Here is an explanation of the code."
    worker_client.complete = AsyncMock(return_value=response)
    worker_client.write_file = AsyncMock()

    worker = WorkerAgent(task=worker_task, client=worker_client)
    result = await worker.run()

    assert result == response
    worker_client.write_file.assert_not_awaited()


@pytest.mark.asyncio
async def test_worker_writes_multiple_files(worker_task: TaskModel, worker_client: MagicMock) -> None:
    response = (
        "FILE: /tmp/f1.py\n```python\ncontent1\n```\n\n"
        "FILE: /tmp/f2.txt\n```\ncontent2\n```"
    )
    worker_client.complete = AsyncMock(return_value=response)

    worker = WorkerAgent(task=worker_task, client=worker_client)
    result = await worker.run()

    assert "/tmp/f1.py" in result
    assert "/tmp/f2.txt" in result
    assert worker_client.write_file.await_count == 2


def test_file_block_regex_matches() -> None:
    text = "FILE: src/main.py\n```python\nx = 1\n```"
    matches = list(_FILE_BLOCK_RE.finditer(text))
    assert len(matches) == 1
    assert matches[0].group("path").strip() == "src/main.py"
    assert matches[0].group("content") == "x = 1\n"


# ---------------------------------------------------------------------------
# LLMClient — Ollama provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ollama_provider_dispatch() -> None:
    s = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=s)

    mock_ollama = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "ollama response"
    mock_ollama.chat.completions.create = AsyncMock(return_value=mock_response)
    client._ollama = mock_ollama

    result = await client.complete("hello", system="be helpful", model="llama3")

    assert result == "ollama response"
    mock_ollama.chat.completions.create.assert_awaited_once()
    call = mock_ollama.chat.completions.create.call_args
    assert call.kwargs["model"] == "llama3"


@pytest.mark.asyncio
async def test_unsupported_provider_raises_value_error() -> None:
    s = Settings(provider=Provider.ANTHROPIC, anthropic_api_key="test-key")
    client = LLMClient(settings=s)

    # Patch _complete_anthropic so we don't hit the network
    with patch.object(
        client, "_complete_anthropic", new=AsyncMock(return_value="ok")
    ) as mock_anthropic:
        result = await client.complete("hi", model="claude")
        assert result == "ok"
        mock_anthropic.assert_awaited_once()
