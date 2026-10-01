"""Tests for Furrow core components."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Config / model tests
# ---------------------------------------------------------------------------

def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_task_default_fields():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_test_result_defaults():
    t = TestResult(passed=False, summary="broken")
    assert t.failures == []


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------

def _make_orchestrator(goal="test goal", max_cycles=0, max_parallel_tasks=5):
    """Create an orchestrator with a mocked LLM client."""
    client = MagicMock()
    client.settings = MagicMock()
    client.settings.max_cycles = max_cycles
    client.settings.max_parallel_tasks = max_parallel_tasks
    return Orchestrator(goal=goal, client=client)


@pytest.mark.asyncio
async def test_is_done_returns_false_without_test_result():
    """When no test result exists and tasks exist, not done."""
    orch = _make_orchestrator()
    orch._current_plan = Plan(
        tasks=[TaskModel(id="1", description="task")], rationale="plan"
    )
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_returns_true_when_tests_pass():
    """When tests pass, orchestrator should report done."""
    orch = _make_orchestrator()
    orch._current_plan = Plan(
        tasks=[TaskModel(id="1", description="task")], rationale="plan"
    )
    orch._test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_returns_false_when_tests_fail():
    """When tests fail, orchestrator should not report done."""
    orch = _make_orchestrator()
    orch._current_plan = Plan(
        tasks=[TaskModel(id="1", description="task")], rationale="plan"
    )
    orch._test_result = TestResult(
        passed=False, summary="broken", failures=["test failed"]
    )
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_returns_true_when_no_tasks_planned():
    """Planner returning no tasks means goal is complete."""
    orch = _make_orchestrator()
    orch._current_plan = Plan(tasks=[], rationale="nothing to do")
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_max_cycles_stops_loop():
    """Orchestrator should stop after max_cycles iterations."""
    orch = _make_orchestrator(max_cycles=2)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="task", files=["a.py"])], rationale="r"
        )
    )
    orch._cycle = AsyncMock()

    await orch.run()
    assert orch.cycles == 2
    assert orch._cycle.call_count == 2


@pytest.mark.asyncio
async def test_cycle_increments_on_each_iteration():
    """Cycles should increment each loop iteration."""
    orch = _make_orchestrator(max_cycles=3)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="task", files=["a.py"])], rationale="r"
        )
    )
    orch._cycle = AsyncMock()

    await orch.run()
    assert orch.cycles == 3


@pytest.mark.asyncio
async def test_progress_callback_emits_events():
    """The on_progress callback should receive event names."""
    events = []

    async def capture(event, data=None):
        events.append(event)

    orch = _make_orchestrator(max_cycles=1)
    orch._on_progress = capture
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(
        return_value=Plan(
            tasks=[TaskModel(id="1", description="task", files=["a.py"])], rationale="r"
        )
    )
    orch._cycle = AsyncMock()

    await orch.run()
    assert "start" in events
    assert "max_cycles_reached" in events


@pytest.mark.asyncio
async def test_get_tasks_returns_current_plan_tasks():
    """_get_tasks should return tasks from the current plan."""
    orch = _make_orchestrator()
    plan = Plan(
        tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
        rationale="r",
    )
    orch._current_plan = plan
    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[1].id == "2"


@pytest.mark.asyncio
async def test_get_tasks_empty_when_no_plan():
    """_get_tasks should return empty list when no plan exists."""
    orch = _make_orchestrator()
    assert orch._get_tasks() == []


# ---------------------------------------------------------------------------
# LLMClient tests
# ---------------------------------------------------------------------------

def test_ollama_uses_openai_client_with_base_url():
    """Ollama provider should configure OpenAI client with the correct base URL."""
    from furrow.config import Provider, Settings

    settings = Settings(provider=Provider.OLLAMA)
    from furrow.llm import LLMClient

    client = LLMClient(settings=settings)
    oai = client.ollama
    # The OpenAI client stores base_url; for Ollama it should include /v1
    assert str(oai.base_url).endswith("/v1")
    assert "11434" in str(oai.base_url)


def test_ollama_reuses_openai_client():
    """Ollama and OpenAI should share the same cached client instance."""
    from furrow.config import Provider, Settings

    settings = Settings(provider=Provider.OLLAMA)
    from furrow.llm import LLMClient

    client = LLMClient(settings=settings)
    ollama_client = client.ollama
    # Calling again should return cached instance
    assert client.ollama is ollama_client
