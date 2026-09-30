from __future__ import annotations

import asyncio
from typing import ClassVar

import pytest

import furrow.core.orchestrator as orch_module
from furrow.config import Plan, TaskModel, TestResult


class FakeSettings:
    def __init__(self, max_cycles: int = 0, max_parallel_tasks: int = 5) -> None:
        self.max_cycles = max_cycles
        self.max_parallel_tasks = max_parallel_tasks


def make_task(task_id: str, description: str = "do something", status: str = "pending") -> TaskModel:
    return TaskModel(id=task_id, description=description, status=status)


class FakePlanner:
    plans: ClassVar[list[Plan]] = []

    def __init__(self, client=None, settings=None, **kwargs):
        self.goals = []
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        plan = self.plans[min(self.calls, len(self.plans) - 1)]
        self.calls += 1
        return plan


class FakeWorkerAgent:
    behavior: ClassVar = "complete"
    _in_flight = 0
    _max_in_flight = 0
    _idx = 0

    def __init__(self, task: TaskModel, client=None, settings=None) -> None:
        self.task = task

    async def run(self) -> str:
        cls = FakeWorkerAgent
        cls._in_flight += 1
        cls._max_in_flight = max(cls._max_in_flight, cls._in_flight)
        cls._idx += 1
        my_idx = cls._idx
        try:
            await asyncio.sleep(0.05)
            if cls.behavior == "complete":
                return f"result-{my_idx}"
            if cls.behavior == "raise":
                raise RuntimeError(f"boom-{my_idx}")
            if callable(cls.behavior):
                return cls.behavior(self.task, my_idx)
            return "ok"
        finally:
            cls._in_flight -= 1

    @classmethod
    def reset(cls) -> None:
        cls.behavior = "complete"
        cls._in_flight = 0
        cls._max_in_flight = 0
        cls._idx = 0


class FakeTesterAgent:
    results: ClassVar[list[TestResult]] = []
    _idx = 0

    def __init__(self, client=None, settings=None) -> None:
        self.client = client

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        cls = FakeTesterAgent
        i = cls._idx
        cls._idx += 1
        return cls.results[min(i, len(cls.results) - 1)]

    @classmethod
    def reset(cls) -> None:
        cls.results = []
        cls._idx = 0


@pytest.fixture(autouse=True)
def reset_fakes():
    FakePlanner.plans = []
    FakeWorkerAgent.reset()
    FakeTesterAgent.reset()
    yield
    FakePlanner.plans = []
    FakeWorkerAgent.reset()
    FakeTesterAgent.reset()


@pytest.fixture
def patch_fakes(monkeypatch):
    monkeypatch.setattr(orch_module, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orch_module, "WorkerAgent", FakeWorkerAgent)
    monkeypatch.setattr(orch_module, "TesterAgent", FakeTesterAgent)
    return monkeypatch


def test_is_done_all_completed():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="g", client=object())
    orch.tasks = [make_task("1", status="completed"), make_task("2", status="completed")]
    assert orch._is_done() is True


def test_is_done_with_failure():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="g", client=object())
    orch.tasks = [make_task("1", status="completed"), make_task("2", status="failed")]
    assert orch._is_done() is False


def test_is_done_no_tasks():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="g", client=object())
    orch.tasks = []
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_no_tasks_terminates_run(patch_fakes, monkeypatch):
    from furrow.core.orchestrator import Orchestrator

    monkeypatch.setattr(orch_module, "settings", FakeSettings(max_cycles=0, max_parallel_tasks=5))
    FakePlanner.plans = [Plan(tasks=[], rationale="nothing to do")]
    FakeTesterAgent.results = [TestResult(passed=True, summary="ok", failures=[])]
    orch = Orchestrator(goal="build a cat", client=object())
    await orch.run()
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_max_cycles_enforced(patch_fakes, monkeypatch):
    from furrow.core.orchestrator import Orchestrator

    monkeypatch.setattr(orch_module, "settings", FakeSettings(max_cycles=1, max_parallel_tasks=5))
    FakePlanner.plans = [Plan(tasks=[make_task("1")], rationale="do it")]
    FakeTesterAgent.results = [TestResult(passed=True, summary="ok", failures=[])]
    orch = Orchestrator(goal="build a cat", client=object())
    await orch.run()
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_max_parallel_tasks_respected(patch_fakes, monkeypatch):
    from furrow.core.orchestrator import Orchestrator

    FakeWorkerAgent.behavior = "complete"
    monkeypatch.setattr(
        orch_module, "settings", FakeSettings(max_cycles=2, max_parallel_tasks=2)
    )
    FakePlanner.plans = [Plan(tasks=[make_task(str(i)) for i in range(5)], rationale="parallel")]
    FakeTesterAgent.results = [TestResult(passed=True, summary="ok", failures=[])]
    orch = Orchestrator(goal="build a cat", client=object())
    await orch.run()
    assert FakeWorkerAgent._max_in_flight <= 2
    assert FakeWorkerAgent._idx == 5
    assert all(t.status == "completed" for t in orch.tasks)


@pytest.mark.asyncio
async def test_goal_preserved_on_failure(patch_fakes, monkeypatch):
    from furrow.core.orchestrator import Orchestrator

    monkeypatch.setattr(
        orch_module, "settings", FakeSettings(max_cycles=2, max_parallel_tasks=5)
    )
    FakePlanner.plans = [Plan(tasks=[make_task("1")], rationale="do it")]
    FakeTesterAgent.results = [
        TestResult(passed=False, summary="broken", failures=["broken"]),
        TestResult(passed=True, summary="ok", failures=[]),
    ]

    def _cycle_worker(task: TaskModel, idx: int) -> str:
        if idx == 1:
            raise RuntimeError("transient failure")
        return "done"

    FakeWorkerAgent.behavior = _cycle_worker

    orch = Orchestrator(goal="Build a cat app", client=object())
    await orch.run()
    assert orch.original_goal == "Build a cat app"
    assert orch.goal == "Build a cat app"
    assert orch.planner.goals[0] == "Build a cat app"
    assert orch.planner.goals[1] == "Fix failing tests:\nbroken"
    assert orch.cycles == 2
