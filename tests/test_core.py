import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.workspace == Path.cwd()


def test_settings_api_key_validation_anthropic():
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is required"):
        Settings(provider=Provider.ANTHROPIC, anthropic_api_key=None)


def test_settings_api_key_validation_openai():
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        Settings(provider=Provider.OPENAI, openai_api_key=None)


def test_orchestrator_is_done_no_plan():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_all_completed():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="completed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_has_failed():
    orchestrator = Orchestrator(goal="test")
    orchestrator.current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="completed"),
            TaskModel(id="2", description="b", status="failed"),
        ],
        rationale="ok",
    )
    assert orchestrator._is_done() is False


def test_orchestrator_get_tasks():
    orchestrator = Orchestrator(goal="test")
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    orchestrator.current_plan = plan
    assert orchestrator._get_tasks() == plan.tasks


@pytest.mark.asyncio
async def test_llm_complete_openai():
    client = LLMClient()
    client.settings.provider = Provider.OPENAI
    client.settings.openai_api_key = "test-key"

    mock_response = AsyncMock()
    mock_response.choices = [AsyncMock(message=AsyncMock(content="hello"))]
    client._openai = AsyncMock()
    client._openai.chat.completions.create.return_value = mock_response

    result = await client.complete("test prompt")
    assert result == "hello"
    client._openai.chat.completions.create.assert_called_once()


@pytest.mark.asyncio
async def test_llm_complete_unsupported():
    client = LLMClient()
    client.settings.provider = "unsupported"  # type: ignore
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("test")


@pytest.mark.asyncio
async def test_llm_complete_ollama():
    client = LLMClient()
    client.settings.provider = Provider.OLLAMA
    client.settings.ollama_base_url = "http://localhost:11434"
    client.settings.model = "llama3"

    mock_response = {"response": "ollama says hi"}
    mock_client = AsyncMock()
    mock_client.post.return_value.__aenter__.return_value.json.return_value = mock_response
    mock_client.post.return_value.__aenter__.return_value.raise_for_status.return_value = None

    with patch("furrow.llm.httpx.AsyncClient", return_value=mock_client):
        result = await client.complete("test prompt")
        assert result == "ollama says hi"
