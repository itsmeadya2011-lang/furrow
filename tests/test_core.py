from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from tenacity import RetryError

from furrow.cli.main import main
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# --- Config validation tests ---


def test_plan_missing_rationale():
    with pytest.raises(ValidationError):
        Plan(tasks=[TaskModel(id="1", description="do thing")])


def test_plan_missing_tasks():
    with pytest.raises(ValidationError):
        Plan(tasks=None, rationale="ok")


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.result is None
    assert t.files == []
    assert t.dependencies == []


def test_task_model_missing_id():
    with pytest.raises(ValidationError):
        TaskModel(description="do thing")


def test_settings_defaults(monkeypatch):
    for key in ["FURROW_PROVIDER", "FURROW_MODEL", "FURROW_OLLAMA_BASE_URL"]:
        monkeypatch.delenv(key, raising=False)
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.model == "claude-sonnet-4-20250514"
    assert s.ollama_base_url == "http://localhost:11434"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_provider_enum_values():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


# --- Orchestrator state tests ---


def test_is_done_empty():
    o = Orchestrator("test")
    assert o._is_done() is True


def test_is_done_all_completed():
    o = Orchestrator("test")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert o._is_done() is True


def test_is_done_some_completed():
    o = Orchestrator("test")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert o._is_done() is False


def test_is_done_with_failure():
    o = Orchestrator("test")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert o._is_done() is False


def test_get_tasks_after_assignment():
    o = Orchestrator("test")
    plan = Plan(
        tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
        rationale="test",
    )
    o.tasks = plan.tasks
    assert o._get_tasks() == plan.tasks
    assert len(o._get_tasks()) == 2


@pytest.mark.asyncio
async def test_cycle_stores_tasks():
    o = Orchestrator("test goal")
    plan = Plan(
        tasks=[TaskModel(id="1", description="task one")],
        rationale="test rationale",
    )

    mock_planner = MagicMock()
    mock_planner.plan = AsyncMock(return_value=plan)
    mock_tester = MagicMock()
    mock_tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

    o.planner = mock_planner

    mock_worker_class = MagicMock()
    mock_worker_instance = MagicMock()
    mock_worker_instance.run = AsyncMock(return_value="mock result")
    mock_worker_class.return_value = mock_worker_instance

    with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester):
        with patch("furrow.core.orchestrator.WorkerAgent", mock_worker_class):
            await o._cycle()

    assert o.tasks == plan.tasks
    assert len(o.tasks) == 1


# --- LLM provider tests ---


def test_llm_anthropic_routing():
    client = LLMClient.__new__(LLMClient)
    client.settings = Settings(provider=Provider.ANTHROPIC)
    client._anthropic = MagicMock()
    client._openai = None

    mock_content = MagicMock()
    mock_content.text = "anthropic response"
    mock_response = MagicMock()
    mock_response.content = [mock_content]
    client._anthropic.messages.create = AsyncMock(return_value=mock_response)

    result = client._complete_anthropic("prompt", "system", "model")
    # Note: _complete_anthropic is sync in signature but calls async methods internally via await
    # Actually it's async def, so we need to run it
    import asyncio
    result = asyncio.run(client._complete_anthropic("prompt", "system", "model"))
    assert result == "anthropic response"


def test_llm_openai_routing():
    client = LLMClient.__new__(LLMClient)
    client.settings = Settings(provider=Provider.OPENAI)
    client._anthropic = None
    client._openai = MagicMock()

    mock_message = MagicMock()
    mock_message.content = "openai response"
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    client._openai.chat.completions.create = AsyncMock(return_value=mock_response)

    import asyncio
    result = asyncio.run(client._complete_openai("prompt", "system", "model"))
    assert result == "openai response"


@pytest.mark.asyncio
async def test_llm_ollama_url_construction():
    client = LLMClient.__new__(LLMClient)
    client.settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://custom:11434")
    client._anthropic = None
    client._openai = None

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message": {"content": "ollama response"}}
    mock_response.raise_for_status = MagicMock()

    with patch("furrow.llm.httpx.AsyncClient") as MockClient:
        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client_instance

        result = await client._complete_ollama("test", "system", "llama3")

        assert result == "ollama response"
        mock_client_instance.post.assert_called_once()
        call_args = mock_client_instance.post.call_args
        assert call_args[0][0] == "http://custom:11434/api/chat"


@pytest.mark.asyncio
async def test_llm_complete_routes_to_ollama():
    client = LLMClient.__new__(LLMClient)
    client.settings = Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434")
    client._anthropic = None
    client._openai = None

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"message": {"content": "ollama response"}}
    mock_response.raise_for_status = MagicMock()

    with patch("furrow.llm.httpx.AsyncClient") as MockClient:
        mock_client_instance = MagicMock()
        mock_client_instance.post = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        MockClient.return_value = mock_client_instance

        result = await client.complete("test prompt")
        assert result == "ollama response"


# --- CLI tests ---


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    # Version comes from importlib.metadata; accept installed or fallback value
    assert "0.1.0" in result.output or "unknown" in result.output


# --- structlog integration tests ---


def test_structlog_loggers_exist():
    from furrow.cli.main import logger as cli_logger
    from furrow.config import logger as config_logger
    from furrow.core.orchestrator import logger as orchestrator_logger
    from furrow.llm import logger as llm_logger
    from furrow.web.server import logger as web_logger

    assert cli_logger is not None
    assert config_logger is not None
    assert orchestrator_logger is not None
    assert llm_logger is not None
    assert web_logger is not None
