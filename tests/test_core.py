from __future__ import annotations

import asyncio
import inspect
from typing import Any
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from furrow.web.server import app

# ---------------------------------------------------------------------------
# Existing tests (kept for backward compatibility)
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Orchestration: _is_done / _get_tasks
# ---------------------------------------------------------------------------


def test_is_done_no_tasks():
    orch = Orchestrator(goal="test")
    assert orch.all_tasks == []
    assert orch._is_done() is True


def test_is_done_all_completed():
    orch = Orchestrator(goal="test")
    orch.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_has_failed():
    orch = Orchestrator(goal="test")
    orch.all_tasks = [TaskModel(id="1", description="a", status="failed")]
    assert orch._is_done() is False


def test_is_done_has_pending():
    orch = Orchestrator(goal="test")
    orch.all_tasks = [TaskModel(id="1", description="a", status="pending")]
    assert orch._is_done() is False


def test_get_tasks_returns_all_tasks():
    orch = Orchestrator(goal="test")
    tasks = [TaskModel(id="1", description="a", status="completed")]
    orch.all_tasks = tasks
    assert orch._get_tasks() is tasks


# ---------------------------------------------------------------------------
# Orchestration: max_cycles enforcement
# ---------------------------------------------------------------------------


async def test_max_cycles_enforced_empty_plan():
    """With max_cycles=1 and an empty plan, the loop should stop after 1 cycle."""
    settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="test", settings=settings)
    mock_plan = Plan(tasks=[], rationale="nothing to do")
    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=mock_plan)):
        await orch.run()
    assert orch.cycles == 1


async def test_max_cycles_enforced_with_failures():
    """When tasks keep failing, max_cycles provides a safety exit."""
    settings = Settings(max_cycles=2)
    orch = Orchestrator(goal="test", settings=settings)
    mock_plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="test")
    test_result = TestResult(passed=True, summary="ok")
    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=mock_plan)), patch.object(
        WorkerAgent, "run", new=AsyncMock(side_effect=Exception("boom"))
    ), patch.object(TesterAgent, "run", new=AsyncMock(return_value=test_result)):
        await orch.run()
    assert orch.cycles == 2


async def test_max_cycles_unlimited_by_default():
    """max_cycles=0 means unlimited — goal completes normally with empty plan."""
    orch = Orchestrator(goal="test")
    mock_plan = Plan(tasks=[], rationale="done")
    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=mock_plan)):
        await orch.run()
    assert orch.cycles == 1


async def test_cycles_increment_and_tasks_tracked():
    """A successful single-cycle run increments cycles and stores tasks."""
    orch = Orchestrator(goal="test")
    task = TaskModel(id="1", description="a")
    mock_plan = Plan(tasks=[task], rationale="test")
    test_result = TestResult(passed=True, summary="ok")
    with patch.object(orch.planner, "plan", new=AsyncMock(return_value=mock_plan)), patch.object(
        WorkerAgent, "run", new=AsyncMock(return_value="implemented")
    ), patch.object(TesterAgent, "run", new=AsyncMock(return_value=test_result)):
        await orch.run()
    assert orch.cycles == 1
    assert len(orch.all_tasks) == 1
    assert orch.all_tasks[0].status == "completed"
    assert orch.all_tasks[0].result == "implemented"


# ---------------------------------------------------------------------------
# Orchestration: max_parallel_tasks semaphore
# ---------------------------------------------------------------------------


def test_semaphore_uses_max_parallel_tasks():
    settings = Settings(max_parallel_tasks=10)
    orch = Orchestrator(goal="test", settings=settings)
    assert isinstance(orch._semaphore, asyncio.Semaphore)
    # _value reflects the initial permit count on asyncio.Semaphore
    assert orch._semaphore._value == 10


def test_semaphore_default_value():
    orch = Orchestrator(goal="test")
    assert orch._semaphore._value == 5  # default max_parallel_tasks


# ---------------------------------------------------------------------------
# Orchestration: console injection
# ---------------------------------------------------------------------------


def test_orchestrator_accepts_custom_console():
    from io import StringIO

    from rich.console import Console

    buf = StringIO()
    custom_console = Console(file=buf, no_color=True, width=80)
    orch = Orchestrator(goal="test", console=custom_console)
    assert orch.console is custom_console


def test_orchestrator_default_console():
    orch = Orchestrator(goal="test")
    assert orch.console is not None


# ---------------------------------------------------------------------------
# LLMClient: settings=None fallback & max_tokens parameter
# ---------------------------------------------------------------------------


def test_llm_client_settings_none_falls_back():
    client = LLMClient(settings=None)
    assert client.settings is not None
    assert client.settings.model == "claude-sonnet-4-20250514"


def test_llm_client_default_settings():
    client = LLMClient()
    assert client.settings is not None
    assert client.settings.provider is not None


def test_llm_client_complete_has_max_tokens_param():
    sig = inspect.signature(LLMClient.complete)
    assert "max_tokens" in sig.parameters
    assert sig.parameters["max_tokens"].default == 4096


def test_llm_client_complete_accepts_model_and_max_tokens():
    sig = inspect.signature(LLMClient.complete)
    params = sig.parameters
    assert "model" in params
    assert params["model"].default is None
    assert "max_tokens" in params


async def test_llm_client_ollama_routing():
    """When provider is OLLAMA, complete() should route to _complete_ollama."""
    settings = Settings(provider="ollama", model="phi3")
    client = LLMClient(settings=settings)

    async def fake_ollama(*args: Any, **kwargs: Any) -> str:
        return "ollama response"

    with patch.object(client, "_complete_ollama", new=fake_ollama):
        result = await client.complete("hello", model="phi3", max_tokens=100)
    assert result == "ollama response"


# ---------------------------------------------------------------------------
# TesterAgent: settings property
# ---------------------------------------------------------------------------


def test_tester_settings_property():
    tester = TesterAgent()
    assert isinstance(tester.settings, Settings)
    assert tester.settings.tester_model is not None


def test_tester_settings_with_explicit_settings():
    settings = Settings(tester_model="gpt-4")
    tester = TesterAgent(settings=settings)
    assert tester.settings.tester_model == "gpt-4"


# ---------------------------------------------------------------------------
# Web server: endpoints
# ---------------------------------------------------------------------------


def test_health_endpoint():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_start_endpoint_returns_response():
    client = TestClient(app)
    with patch("furrow.web.server.Orchestrator") as mock_cls:
        mock_instance = mock_cls.return_value
        mock_instance.run = AsyncMock()
        resp = client.post("/start", json={"goal": "test goal"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["goal"] == "test goal"
    assert data["status"] == "complete"
    assert "output" in data


def test_index_returns_html():
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
