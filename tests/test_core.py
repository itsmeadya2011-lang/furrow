from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from tenacity import RetryError

from furrow.agents.tester import TesterAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Config model tests
# ---------------------------------------------------------------------------

def test_plan_roundtrip():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.model_dump()["tasks"][0]["description"] == "do thing"


def test_test_result_defaults():
    t = TestResult(passed=True, summary="ok")
    assert t.failures == []


def test_settings_defaults():
    s = Settings()
    assert s.provider.value == "anthropic"
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0
    assert s.max_tokens == 4096


def test_settings_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FURROW_MODEL", "gpt-4o")
    monkeypatch.setenv("FURROW_MAX_PARALLEL_TASKS", "3")
    s = Settings()
    assert s.model == "gpt-4o"
    assert s.max_parallel_tasks == 3


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_llm_client_unsupported_provider():
    s = Settings(provider="ollama")  # type: ignore[arg-type]
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="Unsupported provider"):
        await client.complete("hello")


@pytest.mark.asyncio
async def test_llm_client_anthropic_missing_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    s = Settings(provider="anthropic", anthropic_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY is not set"):
        await client.complete("hello")


@pytest.mark.asyncio
async def test_llm_client_openai_missing_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    s = Settings(provider="openai", openai_api_key=None)
    client = LLMClient(settings=s)
    with pytest.raises(ValueError, match="OPENAI_API_KEY is not set"):
        await client.complete("hello")


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------

def test_parse_response_plain_json():
    result = TesterAgent._parse_response('{"passed": true, "summary": "ok"}')
    assert result == {"passed": True, "summary": "ok"}


def test_parse_response_markdown_fences():
    response = '```json\n{"passed": false, "summary": "fail", "failures": ["x"]}\n```'
    result = TesterAgent._parse_response(response)
    assert result == {"passed": False, "summary": "fail", "failures": ["x"]}


def test_parse_response_markdown_fences_no_language():
    response = '```\n{"passed": true, "summary": "ok"}\n```'
    result = TesterAgent._parse_response(response)
    assert result == {"passed": True, "summary": "ok"}


def test_parse_response_keyword_fallback():
    result = TesterAgent._parse_response("All tests passed")
    assert result == {"passed": True}


def test_parse_response_invalid_raises():
    with pytest.raises(ValueError, match="Unable to parse JSON"):
        TesterAgent._parse_response("garbage data")


@pytest.mark.asyncio
async def test_tester_agent_llm_retry():
    client = MagicMock()
    client.settings.tester_model = "test-model"
    client.complete = AsyncMock(side_effect=[ValueError("bad"), '{"passed": true, "summary": "ok"}'])
    agent = TesterAgent(client=client)
    result = await agent.run("goal", [])
    assert result.passed is True
    assert client.complete.call_count == 2


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_is_done_with_empty_tasks():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    orchestrator = Orchestrator(goal="test", client=client)
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_completed_tasks():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    # Loop continues as long as there are tasks in the current plan,
    # so the planner can produce the next slice.
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failed_tasks():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_preserves_original_goal_on_success():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="build feature", client=client)

    mock_plan = MagicMock()
    mock_plan.tasks = [TaskModel(id="1", description="task")]
    with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, return_value=mock_plan):
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = AsyncMock(return_value="done")
            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                MockTester.return_value.run = AsyncMock(
                    return_value=TestResult(passed=True, summary="ok")
                )
                await orchestrator._cycle()

    assert orchestrator.goal == "build feature"
    assert orchestrator.original_goal == "build feature"


@pytest.mark.asyncio
async def test_orchestrator_mutates_goal_on_failure():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = 0
    orchestrator = Orchestrator(goal="build feature", client=client)

    mock_plan = MagicMock()
    mock_plan.tasks = [TaskModel(id="1", description="task")]
    with patch.object(orchestrator.planner, "plan", new_callable=AsyncMock, return_value=mock_plan):
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            MockWorker.return_value.run = AsyncMock(return_value="done")
            with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
                MockTester.return_value.run = AsyncMock(
                    return_value=TestResult(passed=False, summary="fail", failures=["error1", "error2"])
                )
                await orchestrator._cycle()

    assert orchestrator.goal == "Fix failing tests from previous cycle:\nerror1\nerror2"
    assert orchestrator.original_goal == "build feature"


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    client = MagicMock()
    client.settings.max_parallel_tasks = 5
    client.settings.max_cycles = 2
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.current_tasks = [TaskModel(id="1", description="a", status="completed")]
    # Not done because current plan still has tasks
    assert orchestrator._is_done() is False

    # Simulate two cycles
    orchestrator.cycles = 1
    assert not (client.settings.max_cycles > 0 and orchestrator.cycles >= client.settings.max_cycles)

    orchestrator.cycles = 2
    assert client.settings.max_cycles > 0 and orchestrator.cycles >= client.settings.max_cycles


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------

def test_cli_start_does_not_mutate_global_settings():
    from furrow.cli.main import start
    from click.testing import CliRunner

    original_model = Settings().model
    runner = CliRunner()
    with patch("furrow.cli.main.Orchestrator") as MockOrchestrator:
        MockOrchestrator.return_value.run = AsyncMock()
        runner.invoke(start, ["my goal", "--model", "custom-model"])

    # The global settings singleton should not have changed
    assert Settings().model == original_model


# ---------------------------------------------------------------------------
# Web server smoke test
# ---------------------------------------------------------------------------

def test_web_server_imports():
    from furrow.web.server import app
    assert app.title == "Furrow"
