from __future__ import annotations

import pytest

import furrow.core.orchestrator as orch_mod
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeLLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return "fake worker output"


class FakePlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        return self._plan


class FakeTester:
    def __init__(self, result: TestResult | None = None) -> None:
        self.result = result or TestResult(passed=True, summary="ok", failures=[])
        self.calls = 0

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.calls += 1
        return self.result


@pytest.fixture
def client():
    return FakeLLMClient()


def test_get_tasks_initially_empty(client):
    orch = Orchestrator("goal", client=client)
    assert orch._get_tasks() == []


def test_merge_tasks_adds_tasks(client):
    orch = Orchestrator("goal", client=client)
    t1 = TaskModel(id="a", description="a")
    t2 = TaskModel(id="b", description="b")
    orch._merge_tasks([t1, t2])
    assert orch._get_tasks() == [t1, t2]


def test_merge_tasks_dedupes_by_id(client):
    orch = Orchestrator("goal", client=client)
    orch._merge_tasks([TaskModel(id="a", description="a", status="pending")])
    updated = TaskModel(id="a", description="a", status="completed")
    orch._merge_tasks([updated])
    tasks = orch._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == "completed"


def test_is_done_false_when_failed_task(client):
    orch = Orchestrator("goal", client=client)
    orch._merge_tasks([TaskModel(id="a", description="a", status="failed")])
    assert orch._is_done() is False


def test_is_done_true_when_all_completed(client):
    orch = Orchestrator("goal", client=client)
    orch._merge_tasks(
        [
            TaskModel(id="a", description="a", status="completed"),
            TaskModel(id="b", description="b", status="completed"),
        ]
    )
    assert orch._is_done() is True


def test_is_done_true_when_last_plan_empty(client):
    orch = Orchestrator("goal", client=client)
    orch.last_plan = Plan(tasks=[], rationale="nothing to do")
    assert orch._is_done() is True


def test_is_done_false_no_plan_no_tasks(client):
    orch = Orchestrator("goal", client=client)
    assert orch.last_plan is None
    assert orch._get_tasks() == []
    assert orch._is_done() is False


async def test_run_honors_max_cycles_one(client, monkeypatch):
    client.settings = Settings(max_cycles=1)
    plan = Plan(
        tasks=[TaskModel(id="a", description="a")],
        rationale="r",
    )
    orch = Orchestrator("goal", client=client)
    orch.planner = FakePlanner(plan)
    tester = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client: tester)
    await orch.run()
    assert orch.planner.calls == 1
    assert orch.cycles == 1
    assert tester.calls == 1


async def test_run_empty_plan_terminates(client, monkeypatch):
    orch = Orchestrator("goal", client=client)
    orch.planner = FakePlanner(Plan(tasks=[], rationale="nothing"))
    tester = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client: tester)
    await orch.run()
    assert orch.planner.calls == 1
    assert orch._is_done() is True
    assert tester.calls == 0
