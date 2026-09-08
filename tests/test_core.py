import pytest
from unittest.mock import AsyncMock, patch
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_no_plan():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator(goal="test")
    assert o._is_done() is False


def test_orchestrator_is_done_no_tasks():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator(goal="test")
    o.plan = Plan(tasks=[], rationale="empty")
    assert o._is_done() is False


def test_orchestrator_is_done_failed_task():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator(goal="test")
    o.plan = Plan(tasks=[TaskModel(id="1", description="t", status="failed")], rationale="ok")
    assert o._is_done() is False


def test_orchestrator_is_done_completed_tasks():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator(goal="test")
    o.plan = Plan(tasks=[TaskModel(id="1", description="t", status="completed")], rationale="ok")
    assert o._is_done() is True


def test_orchestrator_is_done_mixed_tasks():
    from furrow.core.orchestrator import Orchestrator
    o = Orchestrator(goal="test")
    o.plan = Plan(tasks=[
        TaskModel(id="1", description="t1", status="completed"),
        TaskModel(id="2", description="t2", status="pending"),
    ], rationale="ok")
    assert o._is_done() is False


@pytest.mark.asyncio
async def test_llm_client_provider_ollama():
    with patch("furrow.llm.AsyncOpenAI") as MockOpenAI:
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=AsyncMock(choices=[AsyncMock(message=AsyncMock(content="ollama result"))]))
        MockOpenAI.return_value = mock_client
        from furrow.llm import LLMClient
        s = Settings(provider=Provider.OLLAMA)
        client = LLMClient(settings=s)
        result = await client.complete("hi")
        assert result == "ollama result"
