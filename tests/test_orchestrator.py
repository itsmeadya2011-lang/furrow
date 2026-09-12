"""Tests for the Orchestrator loop logic."""
from __future__ import annotations

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeClient:
    """Minimal LLMClient stand-in for orchestrator tests."""

    class _Settings:
        max_cycles = 0
        workspace = None

    def __init__(self) -> None:
        self.settings = self._Settings()
        self.planner_model = "test"
        self.worker_model = "test"
        self.tester_model = "test"


class FakePlanner:
    def __init__(self, client=None) -> None:
        self.client = client
        self.call_count = 0
        self._plan = Plan(tasks=[], rationale="")

    def set_plan(self, plan: Plan) -> None:
        self._plan = plan

    async def plan(self, goal: str) -> Plan:
        self.call_count += 1
        return self._plan


class FakeWorker:
    def __init__(self, task, client=None, settings=None) -> None:
        self.task = task
        self.client = client
        self.result = "ok"

    async def run(self) -> str:
        self.task.status = "completed"
        return self.result


class FakeTester:
    def __init__(self, client=None, settings=None) -> None:
        self.client = client
        self._result = TestResult(passed=True, summary="all good", failures=[])
        self.call_count = 0

    def set_result(self, result: TestResult) -> None:
        self._result = result

    async def run(self, goal, tasks) -> TestResult:
        self.call_count += 1
        return self._result


def _make_orchestrator(monkeypatch, plan=None, test_result=None):
    client = FakeClient()
    orch = Orchestrator(goal="test goal", client=client)

    planner = FakePlanner(client=client)
    if plan:
        planner.set_plan(plan)
    orch.planner = planner

    if test_result:
        fake_tester = FakeTester(client=client)
        fake_tester.set_result(test_result)
        monkeypatch.setattr(
            "furrow.core.orchestrator.TesterAgent",
            lambda client=None, settings=None: fake_tester,
        )

    monkeypatch.setattr(
        "furrow.core.orchestrator.WorkerAgent",
        lambda task, client=None, settings=None: FakeWorker(task, client),
    )
    return orch, planner


@pytest.mark.asyncio
async def test_no_tasks_halts(monkeypatch):
    orch, planner = _make_orchestrator(
        monkeypatch, plan=Plan(tasks=[], rationale="done")
    )
    await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_all_tasks_pass_halts(monkeypatch):
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task 1"),
            TaskModel(id="2", description="task 2"),
        ],
        rationale="two tasks",
    )
    orch, planner = _make_orchestrator(
        monkeypatch,
        plan=plan,
        test_result=TestResult(passed=True, summary="all passed", failures=[]),
    )
    await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_failed_tests_trigger_next_cycle(monkeypatch):
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="one task",
    )
    orch, planner = _make_orchestrator(
        monkeypatch,
        plan=plan,
        test_result=TestResult(
            passed=False, summary="broken", failures=["test_x failed"]
        ),
    )
    # Planner should be called twice (two cycles)
    await orch.run()
    assert orch.cycles == 2
    assert planner.call_count == 2
    assert "Fix failing tests" in orch.goal


@pytest.mark.asyncio
async def test_max_cycles_enforced(monkeypatch):
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="one task",
    )
    orch, planner = _make_orchestrator(
        monkeypatch,
        plan=plan,
        test_result=TestResult(
            passed=False, summary="broken", failures=["test_x failed"]
        ),
    )
    orch.client.settings.max_cycles = 2
    await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_failed_task_prevents_done(monkeypatch):
    plan = Plan(
        tasks=[TaskModel(id="1", description="task 1")],
        rationale="one task",
    )

    class FailingWorker:
        def __init__(self, task, client=None, settings=None) -> None:
            self.task = task

        async def run(self) -> str:
            self.task.status = "failed"
            raise RuntimeError("boom")

    client = FakeClient()
    orch = Orchestrator(goal="test", client=client)
    planner = FakePlanner(client=client)
    planner.set_plan(plan)
    orch.planner = planner

    monkeypatch.setattr(
        "furrow.core.orchestrator.WorkerAgent",
        lambda task, client=None, settings=None: FailingWorker(task, client),
    )
    monkeypatch.setattr(
        "furrow.core.orchestrator.TesterAgent",
        lambda client=None, settings=None: FakeTester(client=client),
    )

    await orch.run()
    assert orch._is_done() is False
    assert orch._all_tasks[0].status == "failed"


def test_get_tasks_returns_stored():
    orch = Orchestrator(goal="test")
    assert orch._get_tasks() == []
    task = TaskModel(id="1", description="x")
    orch._all_tasks.append(task)
    assert orch._get_tasks() == [task]


def test_done_with_empty_tasks():
    orch = Orchestrator(goal="test")
    assert orch._is_done() is False