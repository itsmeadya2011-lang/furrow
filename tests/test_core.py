import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from openai import AsyncOpenAI

from furrow.agents.planner import PlannerAgent, _extract_json as planner_extract_json
from furrow.agents.tester import _extract_json as tester_extract_json
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def _make_client(provider=Provider.OPENAI):
    client = MagicMock(spec=LLMClient)
    client.settings = Settings(provider=provider, max_cycles=0)
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "ok"}')
    return client


def test_is_done_no_tasks():
    orchestrator = Orchestrator(goal="test", client=_make_client())
    orchestrator.plan = None
    assert orchestrator._is_done() is False


def test_is_done_pending_tasks():
    orchestrator = Orchestrator(goal="test", client=_make_client())
    orchestrator.plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="pending")],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_is_done_all_completed():
    orchestrator = Orchestrator(goal="test", client=_make_client())
    orchestrator.plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="completed")],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


def test_is_done_any_failed():
    orchestrator = Orchestrator(goal="test", client=_make_client())
    orchestrator.plan = Plan(
        tasks=[
            TaskModel(id="1", description="do thing", status="completed"),
            TaskModel(id="2", description="other thing", status="failed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_get_tasks_returns_plan_tasks():
    orchestrator = Orchestrator(goal="test", client=_make_client())
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    orchestrator.plan = plan
    assert orchestrator._get_tasks() == plan.tasks


def test_get_tasks_no_plan():
    orchestrator = Orchestrator(goal="test", client=_make_client())
    orchestrator.plan = None
    assert orchestrator._get_tasks() == []


@pytest.mark.asyncio
async def test_max_cycles_enforcement():
    client = _make_client()
    client.settings = Settings(provider=Provider.OPENAI, max_cycles=2)
    orchestrator = Orchestrator(goal="test", client=client)

    call_count = 0

    async def fake_cycle():
        nonlocal call_count
        call_count += 1

    with patch.object(orchestrator, "_cycle", side_effect=fake_cycle), patch.object(
        orchestrator, "_is_done", return_value=False
    ):
        await orchestrator.run()

    assert call_count == 2


def test_extract_json_plain():
    payload = '{"tasks": [], "rationale": "ok"}'
    assert planner_extract_json(payload) == payload
    assert tester_extract_json(payload) == payload


def test_extract_json_markdown_fence_with_language():
    text = '```json\n{"tasks": [], "rationale": "ok"}\n```'
    expected = '{"tasks": [], "rationale": "ok"}'
    assert planner_extract_json(text) == expected
    assert tester_extract_json(text) == expected


def test_extract_json_markdown_fence_without_language():
    text = '```\n{"tasks": [], "rationale": "ok"}\n```'
    expected = '{"tasks": [], "rationale": "ok"}'
    assert planner_extract_json(text) == expected
    assert tester_extract_json(text) == expected


@pytest.mark.asyncio
async def test_ollama_provider_routing():
    mock_ollama = AsyncMock(spec=AsyncOpenAI)
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="hello"))]
    mock_ollama.chat.completions.create = AsyncMock(return_value=mock_response)

    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    client._ollama = mock_ollama

    result = await client.complete("hello", model="llama3")
    assert result == "hello"
    mock_ollama.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_planner_invalid_json_raises_error():
    client = _make_client()
    client.complete = AsyncMock(return_value="not valid json")
    planner = PlannerAgent(client=client)

    with pytest.raises(ValueError, match="Failed to parse plan from LLM"):
        await planner.plan("test")
