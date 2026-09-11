from io import StringIO
from types import SimpleNamespace

import pytest
from rich.console import Console

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _client(max_cycles: int = 0):
    return SimpleNamespace(settings=SimpleNamespace(max_cycles=max_cycles))


def _plan(*task_ids: str) -> Plan:
    return Plan(
        tasks=[TaskModel(id=task_id, description=task_id) for task_id in task_ids],
        rationale="test plan",
    )


class QueuePlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = list(plans)
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        return self.plans.pop(0)


class FailThenPassWorker:
    def __init__(self, task: TaskModel, client) -> None:
        self.task = task

    async def run(self) -> str:
        if self.task.id == "fail":
            raise RuntimeError("worker failed")
        return "fixed"


class AlwaysFailWorker:
    def __init__(self, task: TaskModel, client) -> None:
        self.task = task

    async def run(self) -> str:
        raise RuntimeError("worker failed")


class PassingWorker:
    def __init__(self, task: TaskModel, client) -> None:
        self.task = task

    async def run(self) -> str:
        return "completed"


class FailThenPassTester:
    def __init__(self, client) -> None:
        self.calls = 0

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.calls += 1
        if self.calls == 1:
            return TestResult(passed=False, summary="failed", failures=["first failure"])
        return TestResult(passed=True, summary="passed", failures=[])


class PassingTester:
    def __init__(self, client) -> None:
        pass

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return TestResult(passed=True, summary="passed", failures=[])


def test_is_done_uses_persisted_task_states():
    orchestrator = Orchestrator("goal", client=_client())

    assert orchestrator._is_done() is True

    orchestrator.tasks = [TaskModel(id="pending", description="pending")]
    assert orchestrator._is_done() is False

    orchestrator.tasks[0].status = "failed"
    assert orchestrator._is_done() is False

    orchestrator.tasks[0].status = "completed"
    assert orchestrator._is_done() is False

    orchestrator.test_result = TestResult(passed=True, summary="passed", failures=[])
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_failed_cycle_continues_until_passing_cycle(monkeypatch):
    orchestrator = Orchestrator("goal", client=_client())
    orchestrator.planner = QueuePlanner([_plan("fail"), _plan("fix")])
    monkeypatch.setattr(
        "furrow.core.orchestrator.WorkerAgent", FailThenPassWorker
    )
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", PassingTester)

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator.planner.calls == 2
    assert orchestrator.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_failed_test_continues_until_passing_test(monkeypatch):
    orchestrator = Orchestrator("goal", client=_client())
    orchestrator.planner = QueuePlanner([_plan("fix"), _plan("fixed")])
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", PassingWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", FailThenPassTester)

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator.planner.calls == 2
    assert orchestrator.test_result is not None
    assert orchestrator.test_result.passed
    assert orchestrator.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_passing_cycle_stops(monkeypatch):
    orchestrator = Orchestrator("goal", client=_client())
    orchestrator.planner = QueuePlanner([_plan("done")])
    monkeypatch.setattr(
        "furrow.core.orchestrator.WorkerAgent", FailThenPassWorker
    )
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", PassingTester)

    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert orchestrator.planner.calls == 1
    assert orchestrator.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_max_cycles_stops_after_configured_count(monkeypatch):
    orchestrator = Orchestrator("goal", client=_client(max_cycles=2))
    orchestrator.planner = QueuePlanner([_plan("fail"), _plan("still-failing")])
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", AlwaysFailWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", PassingTester)
    output = StringIO()
    monkeypatch.setattr(
        "furrow.core.orchestrator.console",
        Console(file=output, force_terminal=False),
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator.planner.calls == 2
    assert "Reached max cycles (2). Halting." in output.getvalue()
