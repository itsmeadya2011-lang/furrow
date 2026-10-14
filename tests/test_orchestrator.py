from __future__ import annotations

from typing import Any

import pytest

from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class FakeLLMClient(LLMClient):
    """An LLMClient that records calls and returns canned responses.

    Avoids needing API keys while still exercising the real orchestrator
    -> agent wiring.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._anthropic = None
        self._openai = None
        self.completions: list[tuple[str, str]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.completions.append((model or self.settings.model, prompt))
        return ""


def _settings(**overrides: Any) -> Settings:
    defaults = {
        "provider": "anthropic",
        "anthropic_api_key": "test",
    }
    defaults.update(overrides)
    return Settings(**defaults)


async def fake_plan(self, goal: str) -> Plan:
    return Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="ok",
    )


async def fake_worker_run(self) -> str:
    return "done"


async def fake_tester_pass(self, goal: str, tasks: list[TaskModel]) -> TestResult:
    return TestResult(passed=True, summary="all good")


async def fake_tester_fail(self, goal: str, tasks: list[TaskModel]) -> TestResult:
    return TestResult(passed=False, summary="fail", failures=["broken"])


@pytest.fixture
def patch_agents(monkeypatch):
    """Patch the agent methods used by the orchestrator to avoid real LLM calls."""
    monkeypatch.setattr(PlannerAgent, "plan", fake_plan)
    monkeypatch.setattr(WorkerAgent, "run", fake_worker_run)
    yield monkeypatch


def test_is_done_no_tasks():
    settings = _settings()
    orch = Orchestrator(goal="x", client=FakeLLMClient(settings))
    assert orch._is_done() is False


def test_is_done_all_completed():
    settings = _settings()
    orch = Orchestrator(goal="x", client=FakeLLMClient(settings))
    t1 = TaskModel(id="1", description="a")
    t2 = TaskModel(id="2", description="b")
    t1.status = "completed"
    t2.status = "completed"
    orch.all_tasks = [t1, t2]
    assert orch._is_done() is True


def test_is_done_with_failure():
    settings = _settings()
    orch = Orchestrator(goal="x", client=FakeLLMClient(settings))
    t1 = TaskModel(id="1", description="a")
    t1.status = "completed"
    t2 = TaskModel(id="2", description="b")
    t2.status = "failed"
    orch.all_tasks = [t1, t2]
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_cycle_stops_on_passing_tests(patch_agents):
    patch_agents.setattr(TesterAgent, "run", fake_tester_pass)
    settings = _settings(max_cycles=10)
    client = FakeLLMClient(settings)

    orch = Orchestrator(goal="build thing", client=client)
    assert await orch._cycle() is True
    assert orch.all_tasks[0].status == "completed"
    assert orch.completed is False
    # Planner, worker, and tester are patched fakes, so the LLM client is never
    # invoked during a cycle.
    assert len(client.completions) == 0


@pytest.mark.asyncio
async def test_cycle_continues_on_failing_tests(patch_agents):
    patch_agents.setattr(TesterAgent, "run", fake_tester_fail)
    settings = _settings(max_cycles=10)
    client = FakeLLMClient(settings)

    orch = Orchestrator(goal="build thing", client=client)
    assert await orch._cycle() is False
    assert orch.goal.startswith("Fix failing tests:")
    assert orch.all_tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_run_respects_max_cycles(patch_agents):
    patch_agents.setattr(TesterAgent, "run", fake_tester_fail)
    settings = _settings(max_cycles=1, max_parallel_tasks=5)
    client = FakeLLMClient(settings)

    orch = Orchestrator(goal="g", client=client)
    await orch.run()
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_run_completes_when_tests_pass(patch_agents):
    patch_agents.setattr(TesterAgent, "run", fake_tester_pass)
    settings = _settings(max_cycles=10)
    client = FakeLLMClient(settings)

    orch = Orchestrator(goal="g", client=client)
    await orch.run()
    assert orch.completed is True
    assert orch.cycles == 1
