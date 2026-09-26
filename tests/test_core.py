"""Unit tests for the Furrow core modules.

These tests avoid hitting any LLM provider by using lightweight fakes.
"""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

# --------------------------------------------------------------------------- #
# Config / model parsing tests
# --------------------------------------------------------------------------- #


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.files == []
    assert t.dependencies == []
    assert t.status == "pending"
    assert t.result is None


def test_plan_json_round_trip():
    data = {
        "tasks": [
            {
                "id": "1",
                "description": "Implement auth",
                "files": ["src/auth.py"],
                "dependencies": [],
            }
        ],
        "rationale": "Auth is the first step",
    }
    plan = Plan(**data)
    assert plan.tasks[0].id == "1"
    assert plan.tasks[0].files == ["src/auth.py"]
    assert plan.rationale == "Auth is the first step"


def test_settings_defaults():
    s = Settings()
    assert s.provider == Provider.ANTHROPIC
    assert s.max_parallel_tasks == 5
    assert s.max_cycles == 0


def test_settings_overrides():
    s = Settings(provider=Provider.OPENAI, max_cycles=3, max_parallel_tasks=2)
    assert s.provider == Provider.OPENAI
    assert s.max_cycles == 3
    assert s.max_parallel_tasks == 2


# --------------------------------------------------------------------------- #
# LLMClient tests
# --------------------------------------------------------------------------- #


def test_llm_client_provider_routing():
    """complete() should delegate to the correct provider method."""
    settings = Settings(provider=Provider.OLLAMA)
    client = LLMClient(settings=settings)

    # Ollama client should be lazily created with the correct base URL.
    assert client._ollama is None
    ollama_client = client.ollama
    assert ollama_client is not None
    assert str(ollama_client.base_url) == str(settings.ollama_base_url) + "/"

    # Verify provider dispatch by mocking the private methods.
    client._complete_anthropic = AsyncMock(return_value="anthropic-result")
    client._complete_openai = AsyncMock(return_value="openai-result")
    client._complete_ollama = AsyncMock(return_value="ollama-result")

    settings.provider = Provider.ANTHROPIC
    result = asyncio.run(client.complete("test"))
    assert result == "anthropic-result"

    settings.provider = Provider.OPENAI
    result = asyncio.run(client.complete("test"))
    assert result == "openai-result"

    settings.provider = Provider.OLLAMA
    result = asyncio.run(client.complete("test"))
    assert result == "ollama-result"


def test_llm_client_unsupported_provider():
    settings = Settings()
    client = LLMClient(settings=settings)
    client.settings = MagicMock(provider="bogus")
    with pytest.raises(ValueError, match="Unsupported provider"):
        asyncio.run(client.complete("test"))


# --------------------------------------------------------------------------- #
# Orchestrator tests
# --------------------------------------------------------------------------- #


class FakePlanner:
    """Fake planner that returns a configurable Plan."""

    def __init__(self, plan: Plan | None = None, exc: Exception | None = None):
        self.plan = AsyncMock(return_value=plan) if plan is not None else None
        self._exc = exc
        self.call_count = 0

    async def plan(self, goal: str) -> Plan:
        self.call_count += 1
        if self._exc is not None:
            raise self._exc
        return await self.plan(goal)  # type: ignore[unreachable]


def _make_orchestrator(plan: Plan | None = None, exc=None, max_cycles: int = 0) -> Orchestrator:
    settings = Settings(max_cycles=max_cycles)
    orch = Orchestrator(goal="Test goal", settings=settings)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan) if plan is not None else None

    # Fake the LLM client so nothing tries to connect.
    orch.client = MagicMock()
    orch.client.settings = settings
    return orch


def _completed_task(task_id: str, desc: str = "task") -> TaskModel:
    t = TaskModel(id=task_id, description=desc)
    t.status = "completed"
    t.result = "done"
    return t


def _failed_task(task_id: str, desc: str = "task") -> TaskModel:
    t = TaskModel(id=task_id, description=desc)
    t.status = "failed"
    t.result = "error"
    return t


@pytest.mark.asyncio
async def test_is_done_no_tasks():
    """When the plan has no tasks, the orchestrator should be done."""
    orch = _make_orchestrator(plan=Plan(tasks=[], rationale="nothing to do"))
    await orch._cycle()
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_all_tasks_completed_and_tests_passed():
    """All tasks completed + tests passed => done."""
    plan = Plan(
        tasks=[_completed_task("1", "build feature")],
        rationale="single task",
    )
    orch = _make_orchestrator(plan=plan)
    # Mock the worker execution so _cycle doesn't call the real LLM.
    with patch("furrow.core.orchestrator.WorkerAgent") as WorkerMock, \
         patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        instance = WorkerMock.return_value
        instance.run = AsyncMock(return_value="implementation result")
        tester_instance = TesterMock.return_value
        tester_instance.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all green", failures=[])
        )
        await orch._cycle()

    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_tasks_failed():
    """Failed tasks mean not done."""
    plan = Plan(
        tasks=[_failed_task("1", "build feature")],
        rationale="single task",
    )
    orch = _make_orchestrator(plan=plan)
    with patch("furrow.core.orchestrator.WorkerAgent") as WorkerMock, \
         patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        instance = WorkerMock.return_value
        instance.run = AsyncMock(return_value="result")
        tester_instance = TesterMock.return_value
        tester_instance.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        # Manually set task status to failed to simulate a crash.
        orch.current_plan = plan
        plan.tasks[0].status = "failed"

    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_tests_failed():
    """Tasks completed but tests failed => not done."""
    plan = Plan(
        tasks=[_completed_task("1", "build feature")],
        rationale="single task",
    )
    orch = _make_orchestrator(plan=plan)
    orch.current_plan = plan
    orch.last_test_result = TestResult(
        passed=False, summary="2 failures", failures=["test_auth", "test_auth2"]
    )

    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_tests_failed_but_tasks_completed():
    """Even with all tasks completed, failing tests mean not done."""
    plan = Plan(
        tasks=[_completed_task("1", "task"), _completed_task("2", "task2")],
        rationale="two tasks",
    )
    orch = _make_orchestrator(plan=plan)
    orch.current_plan = plan
    orch.last_test_result = TestResult(
        passed=False, summary="broken", failures=["fail"]
    )

    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_max_cycles_stop():
    """Orchestrator should stop when max_cycles is reached."""
    plan = Plan(
        tasks=[_completed_task("1", "task")],
        rationale="single task",
    )
    orch = _make_orchestrator(plan=plan, max_cycles=1)

    with patch("furrow.core.orchestrator.WorkerAgent") as WorkerMock, \
         patch("furrow.core.orchestrator.TesterAgent") as TesterMock:
        instance = WorkerMock.return_value
        instance.run = AsyncMock(return_value="result")
        tester_instance = TesterMock.return_value
        tester_instance.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        await orch._cycle()

    assert orch.cycles == 0  # cycles not yet incremented at this point
    assert orch.current_plan is not None
    # After first cycle, _is_done returns True because max_cycles == 1.
    orch.cycles = 1
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_plan_parse_error_handled():
    """A ValueError from the planner should be caught and return False."""
    orch = _make_orchestrator()
    orch.planner.plan = AsyncMock(side_effect=ValueError("Failed to parse plan"))

    cycle_ok = await orch._cycle()
    assert cycle_ok is False
    assert orch._done is False


@pytest.mark.asyncio
async def test_task_accumulation_across_cycles():
    """Tasks should accumulate in all_tasks."""
    plan1 = Plan(tasks=[_completed_task("1")], rationale="cycle 1")
    plan2 = Plan(tasks=[_completed_task("2"), _completed_task("3")], rationale="cycle 2")
    orch = _make_orchestrator(plan=plan1)

    # First cycle
    with patch("furrow.core.orchestrator.WorkerAgent") as WMock, \
         patch("furrow.core.orchestrator.TesterAgent") as TMock:
        WMock.return_value.run = AsyncMock(return_value="r")
        TMock.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["x"])
        )
        await orch._cycle()

    assert len(orch.all_tasks) == 1

    # Second cycle
    orch.planner.plan = AsyncMock(return_value=plan2)
    with patch("furrow.core.orchestrator.WorkerAgent") as WMock, \
         patch("furrow.core.orchestrator.TesterAgent") as TMock:
        WMock.return_value.run = AsyncMock(return_value="r")
        TMock.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        await orch._cycle()

    assert len(orch.all_tasks) == 3


@pytest.mark.asyncio
async def test_goal_rewrite_on_test_failure():
    """When tests fail, the goal should be rewritten for the next cycle."""
    plan = Plan(tasks=[_completed_task("1")], rationale="task")
    orch = _make_orchestrator(plan=plan)
    orch.current_plan = plan

    with patch("furrow.core.orchestrator.WorkerAgent") as WMock, \
         patch("furrow.core.orchestrator.TesterAgent") as TMock:
        WMock.return_value.run = AsyncMock(return_value="r")
        TMock.return_value.run = AsyncMock(
            return_value=TestResult(
                passed=False, summary="2 failures", failures=["failure A", "failure B"]
            )
        )
        await orch._cycle()

    assert orch.goal.startswith("Fix failing tests:")
    assert "failure A" in orch.goal
    assert "failure B" in orch.goal


@pytest.mark.asyncio
async def test_on_event_called():
    """The on_event callback should receive stream of events."""
    events = []

    async def callback(event_type: str, payload: object) -> None:
        events.append((event_type, payload))

    plan = Plan(tasks=[_completed_task("1")], rationale="task")
    orch = Orchestrator(goal="test", settings=Settings(), on_event=callback)
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)
    orch.client = MagicMock()
    orch.client.settings = orch.settings

    with patch("furrow.core.orchestrator.WorkerAgent") as WMock, \
         patch("furrow.core.orchestrator.TesterAgent") as TMock:
        WMock.return_value.run = AsyncMock(return_value="implementation")
        TMock.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )
        await orch._cycle()

    # Should have received multiple event types.
    event_types = [e[0] for e in events]
    assert "cycle_start" in event_types
    assert "plan" in event_types
    assert "execution_start" in event_types
    assert "task_completed" in event_types
    assert "test_passed" in event_types


@pytest.mark.asyncio
async def test_no_max_cycles_means_infinite():
    """With max_cycles=0, the cycle limit should not trigger stopping."""
    plan = Plan(tasks=[_completed_task("1")], rationale="task")
    orch = _make_orchestrator(plan=plan, max_cycles=0)
    orch.current_plan = plan
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])

    # max_cycles=0 means unlimited, so _is_done should only depend on task/test state.
    orch.cycles = 100
    assert orch._is_done() is True  # All tasks done + tests passed


@pytest.mark.asyncio
async def test_partial_completion_not_done():
    """Some tasks pending => not done."""
    plan = Plan(
        tasks=[
            _completed_task("1"),
            TaskModel(id="2", description="pending task"),
        ],
        rationale="two tasks",
    )
    orch = _make_orchestrator(plan=plan)
    orch.current_plan = plan
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])

    assert orch._is_done() is False
