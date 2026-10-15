"""Comprehensive tests for Furrow core modules."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.id == "1"
    assert t.description == "do thing"
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_construction():
    p = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )
    assert p.tasks[0].description == "do thing"
    assert p.rationale == "ok"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
    assert t.summary == "ok"
    assert t.failures == []


def test_provider_enum():
    assert Provider.ANTHROPIC == "anthropic"
    assert Provider.OPENAI == "openai"
    assert Provider.OLLAMA == "ollama"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_cycles == 0
    assert s.max_parallel_tasks == 5
    assert s.ollama_base_url == "http://localhost:11434"


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


def test_orchestrator_is_done_no_tasks():
    """With no tasks, _is_done() should return False (not complete)."""
    o = Orchestrator(goal="test")
    assert o._is_done() is False


def test_orchestrator_is_done_all_completed():
    """When all tasks are completed, _is_done() returns True."""
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="task one", status="completed"),
        TaskModel(id="2", description="task two", status="completed"),
    ]
    assert o._is_done() is True


def test_orchestrator_is_done_some_pending():
    """When some tasks are pending, _is_done() returns False."""
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="task one", status="completed"),
        TaskModel(id="2", description="task two", status="pending"),
    ]
    assert o._is_done() is False


def test_orchestrator_is_done_one_failed():
    """When any task failed, _is_done() returns False."""
    o = Orchestrator(goal="test")
    o._tasks = [
        TaskModel(id="1", description="task one", status="completed"),
        TaskModel(id="2", description="task two", status="failed"),
    ]
    assert o._is_done() is False


def test_orchestrator_get_tasks_returns_tracked_tasks():
    """_get_tasks() returns the internally tracked task list."""
    o = Orchestrator(goal="test")
    t = TaskModel(id="1", description="task one")
    o._tasks.append(t)
    assert o._get_tasks() == [t]


def test_orchestrator_max_cycles_halt():
    """When max_cycles is reached, run() should halt."""
    o = Orchestrator(goal="test", settings=Settings(max_cycles=1))
    # Set cycles to max so the first check triggers halt
    o.cycles = 1
    # Patch _cycle to avoid actual execution
    o._cycle = AsyncMock()
    o._is_done = MagicMock(return_value=False)
    events = []
    o.on_progress = lambda e: events.append(e)
    asyncio.run(o.run())
    # Should have halted due to max_cycles=1 (1 >= 1)
    assert any(e.get("event") == "halted" for e in events)


def test_orchestrator_progress_events():
    """on_progress callback receives lifecycle events."""
    o = Orchestrator(goal="test", settings=Settings(max_cycles=1))
    # Set cycles to max so the first check triggers halt
    o.cycles = 1
    o._cycle = AsyncMock()
    o._is_done = MagicMock(return_value=False)
    events = []
    o.on_progress = lambda e: events.append(e)
    asyncio.run(o.run())
    assert any(e.get("event") == "started" for e in events)
    assert any(e.get("event") == "halted" for e in events)


def test_orchestrator_task_tracking_across_cycles():
    """Tasks planned in one cycle are tracked and status persists."""
    o = Orchestrator(goal="test", settings=Settings(max_cycles=1))
    plan = Plan(
        tasks=[TaskModel(id="1", description="task one")],
        rationale="test",
    )
    o.planner.plan = AsyncMock(return_value=plan)

    async def fake_worker_run():
        return "done"

    with patch.object(PlannerAgent, "plan", new=AsyncMock(return_value=plan)):
        with patch(
            "furrow.core.orchestrator.WorkerAgent",
            return_value=MagicMock(run=AsyncMock(return_value="done")),
        ):
            with patch(
                "furrow.core.orchestrator.TesterAgent",
                return_value=MagicMock(
                    run=AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))
                ),
            ):
                asyncio.run(o.run())

    # After the cycle, the task should be tracked as completed
    assert len(o._tasks) == 1
    assert o._tasks[0].status == "completed"
    assert o._tasks[0].result == "done"


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------


def test_llm_client_provider_selection_anthropic():
    client = LLMClient(settings=Settings(provider=Provider.ANTHROPIC))
    assert client.settings.provider == Provider.ANTHROPIC


def test_llm_client_provider_selection_openai():
    client = LLMClient(settings=Settings(provider=Provider.OPENAI))
    assert client.settings.provider == Provider.OPENAI


def test_llm_client_provider_selection_ollama():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA))
    assert client.settings.provider == Provider.OLLAMA


def test_llm_client_ollama_property():
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"))
    ollama_client = client.ollama
    assert ollama_client is not None
    # Second call returns the same cached client
    assert client.ollama is ollama_client


@pytest.mark.asyncio
async def test_llm_client_complete_ollama_unsupported_without_server():
    """Ollama complete() should work with the OpenAI-compatible client (may fail at network level, not provider level)."""
    client = LLMClient(settings=Settings(provider=Provider.OLLAMA, ollama_base_url="http://localhost:11434"))
    # The provider is recognized (no ValueError for "unsupported provider")
    # It may fail connecting to Ollama, but that's a runtime/network error
    try:
        await client.complete("Say hi", model="llama3")
    except ValueError as e:
        pytest.fail(f"Ollama provider should be supported, got: {e}")
    except Exception:
        # Network/connection errors are expected without a running Ollama server
        pass


# ---------------------------------------------------------------------------
# PlannerAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planner_agent_parses_valid_json():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        return_value=json.dumps(
            {
                "tasks": [
                    {"id": "1", "description": "Implement auth", "files": ["auth.py"], "dependencies": []}
                ],
                "rationale": "Auth is needed first",
            }
        )
    )
    client.settings.planner_model = "test-model"
    planner = PlannerAgent(client=client)
    plan = await planner.plan("Add auth")
    assert len(plan.tasks) == 1
    assert plan.tasks[0].description == "Implement auth"
    assert plan.rationale == "Auth is needed first"


@pytest.mark.asyncio
async def test_planner_agent_raises_on_invalid_json():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value="not json at all")
    client.settings.planner_model = "test-model"
    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError, match="Failed to parse plan"):
        await planner.plan("Add auth")


# ---------------------------------------------------------------------------
# TesterAgent tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tester_agent_parses_valid_json():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(
        return_value=json.dumps(
            {"passed": True, "summary": "All tests passed", "failures": []}
        )
    )
    client.settings.tester_model = "test-model"
    tester = TesterAgent(client=client)
    result = await tester.run("test goal", [TaskModel(id="1", description="task")])
    assert result.passed is True
    assert result.summary == "All tests passed"
    assert result.failures == []


@pytest.mark.asyncio
async def test_tester_agent_falls_back_on_invalid_json():
    client = MagicMock(spec=LLMClient)
    client.complete = AsyncMock(return_value="tests passed everything looks good")
    client.settings.tester_model = "test-model"
    tester = TesterAgent(client=client)
    result = await tester.run("test goal", [TaskModel(id="1", description="task")])
    # Fallback: "passed" in response.lower() -> True
    assert result.passed is True


@pytest.mark.asyncio
async def test_tester_agent_run_tests_exception():
    client = MagicMock(spec=LLMClient)
    client.settings.tester_model = "test-model"
    tester = TesterAgent(client=client)
    with patch.object(tester, "_run_tests", side_effect=RuntimeError("boom")):
        result = await tester.run("test goal", [])
        assert result.passed is False
        assert "boom" in result.summary
