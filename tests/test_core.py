import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_no_tasks():
    settings = Settings(max_cycles=1)
    orchestrator = Orchestrator(goal="test", settings=settings)
    orchestrator.current_plan = Plan(tasks=[], rationale="none")
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_all_completed():
    settings = Settings(max_cycles=1)
    orchestrator = Orchestrator(goal="test", settings=settings)
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_has_failed():
    settings = Settings(max_cycles=1)
    orchestrator = Orchestrator(goal="test", settings=settings)
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_partial_completed():
    settings = Settings(max_cycles=1)
    orchestrator = Orchestrator(goal="test", settings=settings)
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="pending"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_get_tasks():
    settings = Settings(max_cycles=1)
    orchestrator = Orchestrator(goal="test", settings=settings)
    assert orchestrator._get_tasks() == []
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    orchestrator.current_plan = plan
    assert orchestrator._get_tasks() == plan.tasks


@pytest.mark.asyncio
async def test_orchestrator_max_cycles():
    settings = Settings(max_cycles=2)
    orchestrator = Orchestrator(goal="test", settings=settings)
    orchestrator.cycles = 2

    with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=Plan(tasks=[], rationale="done")):
        await orchestrator.run()

    assert orchestrator.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_full_cycle():
    settings = Settings(max_cycles=1)
    orchestrator = Orchestrator(goal="test", settings=settings)

    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", files=["test.py"])],
        rationale="plan",
    )

    with patch.object(PlannerAgent, "plan", new_callable=AsyncMock, return_value=plan):
        with patch.object(WorkerAgent, "run", new_callable=AsyncMock, return_value="done"):
            with patch.object(
                TesterAgent,
                "run",
                new_callable=AsyncMock,
                return_value=TestResult(passed=True, summary="ok", failures=[]),
            ):
                await orchestrator.run()

    assert orchestrator.cycles == 1
    assert plan.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_llm_client_ollama():
    from furrow.llm import LLMClient

    settings = Settings(provider="ollama", ollama_base_url="http://localhost:11434")
    client = LLMClient(settings=settings)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="ollama response"))]

    with patch("furrow.llm.AsyncOpenAI") as mock_openai:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai.return_value = mock_client

        result = await client.complete("hello", model="llama3")
        assert result == "ollama response"
        mock_openai.assert_called_once_with(base_url="http://localhost:11434/v1", api_key="ollama")
        mock_client.chat.completions.create.assert_called_once()
