from typing import Any

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module


class FakePlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = plans
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        if not self.plans:
            raise AssertionError("planner called too many times")
        return self.plans.pop(0)


class FakeTester:
    def __init__(self, result: TestResult) -> None:
        self.result = result
        self.calls = 0
        self.task_ids: list[list[str]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.calls += 1
        self.task_ids.append([task.id for task in tasks])
        return self.result


class StaticWorker:
    def __init__(self, task: TaskModel, client: Any = None, settings: Any = None) -> None:
        self.task = task

    async def run(self) -> str:
        return f"done {self.task.id}"


class FailingWorker:
    def __init__(self, task: TaskModel, client: Any = None, settings: Any = None) -> None:
        self.task = task

    async def run(self) -> str:
        raise RuntimeError("worker failed")


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


async def test_get_tasks_returns_current_plan_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskModel(id="1", description="do thing")
    plan = Plan(tasks=[task], rationale="ok")
    planner: Any = FakePlanner([plan])
    tester: Any = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    orch = orchestrator_module.Orchestrator("goal", planner=planner, tester=tester)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", StaticWorker)

    await orch._cycle()

    assert orch._get_tasks() is plan.tasks
    assert orch._get_tasks()[0].status == "completed"


async def test_all_completed_passing_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskModel(id="1", description="do thing")
    plan = Plan(tasks=[task], rationale="ok")
    planner: Any = FakePlanner([plan])
    tester: Any = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    orch = orchestrator_module.Orchestrator("goal", planner=planner, tester=tester)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", StaticWorker)

    await orch._cycle()

    assert orch._is_done() is True


async def test_failed_tests_continue(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskModel(id="1", description="do thing")
    plan = Plan(tasks=[task], rationale="ok")
    planner: Any = FakePlanner([plan])
    tester: Any = FakeTester(TestResult(passed=False, summary="bad", failures=["test failed"]))
    orch = orchestrator_module.Orchestrator("goal", planner=planner, tester=tester)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", StaticWorker)

    await orch._cycle()

    assert orch._is_done() is False
    assert orch.goal.startswith("Fix failing tests:")


async def test_failed_worker_task_does_not_stop(monkeypatch: pytest.MonkeyPatch) -> None:
    task = TaskModel(id="1", description="do thing")
    plan = Plan(tasks=[task], rationale="ok")
    planner: Any = FakePlanner([plan])
    tester: Any = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    orch = orchestrator_module.Orchestrator("goal", planner=planner, tester=tester)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FailingWorker)

    await orch._cycle()

    assert orch._get_tasks()[0].status == "failed"
    assert orch._is_done() is False


async def test_empty_plan_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = Plan(tasks=[], rationale="done")
    planner: Any = FakePlanner([plan])
    tester: Any = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    orch = orchestrator_module.Orchestrator("goal", planner=planner, tester=tester)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", StaticWorker)

    await orch._cycle()

    assert orch._is_done() is True
    assert orch._get_tasks() == []
    assert tester.calls == 0


async def test_max_cycles_stops_after_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    task = TaskModel(id="1", description="do thing")
    plan = Plan(tasks=[task], rationale="ok")
    planner: Any = FakePlanner([plan, plan])
    tester: Any = FakeTester(TestResult(passed=False, summary="bad", failures=["test failed"]))
    orch = orchestrator_module.Orchestrator("goal", planner=planner, tester=tester)
    monkeypatch.setattr(orch.client.settings, "max_cycles", 2)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", StaticWorker)

    await orch.run()

    output = capsys.readouterr().out
    assert orch.cycles == 2
    assert planner.calls == 2
    assert "Max cycles set to 2" in output
    assert "Max cycles reached (2/2)" in output
