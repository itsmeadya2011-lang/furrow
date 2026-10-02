"""Tests for the Furrow orchestrator."""

from __future__ import annotations

from typing import Any

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeLLMClient:
    """Fake LLM client that doesn't make real API calls."""

    def __init__(self) -> None:
        self.settings = Settings()
        self.complete_calls: list[str] = []
        self.tool_calls: list[str] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append(prompt)
        return "done"

    async def complete_with_tools(self, prompt: str, system: str = "", **kwargs: Any) -> str:
        self.tool_calls.append(prompt)
        return "Implemented the task successfully."


class FakePlanner:
    """Planner that returns a predefined plan."""

    def __init__(self, plans: list[Plan]) -> None:
        self._plans = list(plans)
        self.plan_calls: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.plan_calls.append(goal)
        return self._plans.pop(0)


class FakeTester:
    """Tester that returns a predefined result."""

    def __init__(self, results: list[TestResult]) -> None:
        self._results = list(results)
        self.test_calls: list[tuple[str, Any]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.test_calls.append((goal, tasks))
        return self._results.pop(0)


class FakeWorker:
    """Worker that returns a predefined result. Accepts same kwargs as WorkerAgent."""

    def __init__(self, task: TaskModel, client: Any = None, settings: Any = None, **kwargs: Any) -> None:
        self.task = task
        self._result = "completed"
        self.worker_calls: list[str] = []

    async def run(self) -> str:
        self.worker_calls.append("worker ran")
        return self._result


def _patch_worker_and_tester(monkeypatch: pytest.MonkeyPatch, tester_result: TestResult) -> None:
    """Patch WorkerAgent and TesterAgent in the orchestrator module."""
    from furrow.core import orchestrator as orch_module

    monkeypatch.setattr(orch_module, "WorkerAgent", FakeWorker, raising=True)
    # TesterAgent is called as TesterAgent(client=..., settings=...).run(...)
    monkeypatch.setattr(
        orch_module, "TesterAgent",
        lambda client=None, settings=None: FakeTester([tester_result]),
        raising=True,
    )


# --- _is_done tests ---


def test_is_done_no_tasks():
    settings = Settings(max_cycles=0)
    orch = Orchestrator(goal="test", settings=settings)
    orch.tasks = []
    assert orch._is_done() is True


def test_is_done_no_test_result():
    orch = Orchestrator(goal="test", client=FakeLLMClient())
    orch.tasks = [TaskModel(id="1", description="do stuff")]
    orch.last_test_result = None
    assert orch._is_done() is False


def test_is_done_tests_passed():
    orch = Orchestrator(goal="test", client=FakeLLMClient())
    orch.tasks = [TaskModel(id="1", description="do stuff", status="completed")]
    orch.last_test_result = TestResult(passed=True, summary="all good")
    assert orch._is_done() is True


def test_is_done_tests_failed():
    orch = Orchestrator(goal="test", client=FakeLLMClient())
    orch.tasks = [TaskModel(id="1", description="do stuff", status="completed")]
    orch.last_test_result = TestResult(passed=False, summary="broken", failures=["test"])
    assert orch._is_done() is False


# --- max_cycles enforcement ---


@pytest.mark.asyncio
async def test_max_cycles_enforced(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="test", client=FakeLLMClient(), settings=settings)

    orch.planner = FakePlanner([  # type: ignore[assignment]
        Plan(tasks=[TaskModel(id="1", description="task")], rationale="ok"),
    ])
    _patch_worker_and_tester(monkeypatch, TestResult(passed=True, summary="ok"))

    await orch.run()

    # With max_cycles=1, the loop body should execute exactly once
    assert orch.cycles == 1
    assert orch.last_test_result is not None
    assert orch.last_test_result.passed


# --- Task tracking ---


@pytest.mark.asyncio
async def test_tasks_stored_after_cycle(monkeypatch: pytest.MonkeyPatch):
    settings = Settings(max_cycles=0)
    orch = Orchestrator(goal="test", client=FakeLLMClient(), settings=settings)

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task A"),
            TaskModel(id="2", description="task B"),
        ],
        rationale="plan",
    )
    orch.planner = FakePlanner([plan])  # type: ignore[assignment]
    _patch_worker_and_tester(monkeypatch, TestResult(passed=True, summary="ok"))

    await orch.run()

    # Tasks should be stored and marked as completed
    assert len(orch.tasks) == 2
    assert all(t.status == "completed" for t in orch.tasks)


# --- max_parallel_tasks semaphore ---


def test_semaphore_initialized():
    settings = Settings(max_parallel_tasks=3)
    orch = Orchestrator(goal="test", client=FakeLLMClient(), settings=settings)
    # Semaphore is initialized in __init__
    assert orch._semaphore is not None
    assert orch._semaphore._value == 3  # type: ignore[attr-defined]


def test_settings_propagation():
    settings = Settings(max_cycles=5, max_parallel_tasks=2)
    orch = Orchestrator(goal="test", settings=settings)
    assert orch.settings.max_cycles == 5
    assert orch.settings.max_parallel_tasks == 2


def test_client_inherits_settings():
    settings = Settings(max_cycles=10)
    client = FakeLLMClient()
    client.settings = settings
    orch = Orchestrator(goal="test", client=client)
    assert orch.settings is settings
