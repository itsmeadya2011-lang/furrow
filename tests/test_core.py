import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_orchestrator_stores_plan():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings.max_parallel_tasks = 5

    mock_plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="ok")
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "ok"}')

    orchestrator = Orchestrator(goal="test", client=mock_client)
    await orchestrator._cycle()
    assert orchestrator.plan is not None
    assert orchestrator._get_tasks() == []


@pytest.mark.asyncio
async def test_orchestrator_max_parallel_tasks_batches():
    mock_client = MagicMock(spec=LLMClient)
    mock_client.settings.max_parallel_tasks = 2

    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
        TaskModel(id="3", description="task 3"),
    ]
    mock_plan = Plan(tasks=tasks, rationale="ok")
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "ok"}')

    orchestrator = Orchestrator(goal="test", client=mock_client)

    with patch.object(
        orchestrator.planner, "plan", new_callable=AsyncMock, return_value=mock_plan
    ), patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, patch(
        "furrow.core.orchestrator.TesterAgent"
    ) as MockTester:
        MockWorker.return_value.run = AsyncMock(return_value="done")
        mock_tester = MagicMock()
        mock_tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
        MockTester.return_value = mock_tester

        await orchestrator._cycle()

        assert MockWorker.call_count == 3


@pytest.mark.asyncio
async def test_llm_client_ollama():
    with patch.dict(os.environ, {}, clear=True):
        client = LLMClient.__new__(LLMClient)
        client.settings = MagicMock()
        client.settings.provider = "ollama"
        client.settings.ollama_base_url = "http://localhost:11434"
        client.settings.model = "llama3"

        with patch("httpx.AsyncClient") as MockClient:
            mock_response = MagicMock()
            mock_response.json.return_value = {"response": "ollama response"}
            mock_response.raise_for_status = MagicMock()
            mock_client_instance = AsyncMock()
            mock_client_instance.post.return_value = mock_response
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client_instance

            result = await client.complete("test prompt", model="llama3")
            assert result == "ollama response"
            mock_client_instance.post.assert_called_once()
            call_args = mock_client_instance.post.call_args
            assert call_args[0][0] == "http://localhost:11434/api/generate"


@pytest.mark.asyncio
async def test_llm_client_unsupported_provider():
    client = LLMClient.__new__(LLMClient)
    client.settings = MagicMock()
    client.settings.provider = "unknown"

    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test prompt")
