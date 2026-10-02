from __future__ import annotations

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from tests.conftest import FakeClient, make_settings


class FakePlanner:
    def __init__(self, *args, **kwargs):
        pass

    async def plan(self, goal):
        return Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")


class FakeWorker:
    def __init__(self, task, *args, **kwargs):
        self.task = task

    async def run(self):
        return "ok"


class FakeTester:
    def __init__(self, *args, **kwargs):
        pass

    async def run(self, goal, tasks):
        return TestResult(passed=True, summary="ok", failures=[])


async def test_max_cycles_halts_after_one(monkeypatch):
    settings = make_settings(max_cycles=1, max_parallel_tasks=5)
    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", FakePlanner)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", FakeTester)
    client = FakeClient(settings=settings)
    orch = Orchestrator(goal="g", client=client, settings=settings)
    await orch.run()
    assert orch.cycles == 1


async def test_task_tracking_after_run(monkeypatch):
    settings = make_settings(max_cycles=5, max_parallel_tasks=5)
    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", FakePlanner)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", FakeTester)
    client = FakeClient(settings=settings)
    orch = Orchestrator(goal="g", client=client, settings=settings)
    await orch.run()
    tasks = orch._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].status == "completed"
    assert tasks[0].result == "ok"


async def test_is_done_states(monkeypatch):
    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", FakePlanner)
    settings = make_settings(max_parallel_tasks=5)
    orch = Orchestrator(goal="g", client=FakeClient(settings=settings), settings=settings)

    orch._tasks = []
    assert orch._is_done() is True

    orch._tasks = [TaskModel(id="1", description="x", status="completed", result="ok")]
    assert orch._is_done() is True

    orch._tasks = [TaskModel(id="1", description="x", status="failed", result="err")]
    assert orch._is_done() is False

    orch._tasks = [
        TaskModel(id="1", description="x", status="completed", result="ok"),
        TaskModel(id="2", description="y"),
    ]
    assert orch._is_done() is False


class _ThreeTaskPlanner:
    def __init__(self, *args, **kwargs):
        pass

    async def plan(self, goal):
        return Plan(
            tasks=[TaskModel(id=str(i), description=f"task {i}") for i in range(3)],
            rationale="r",
        )


async def test_parallel_tasks_complete(monkeypatch):
    settings = make_settings(max_cycles=0, max_parallel_tasks=2)
    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", _ThreeTaskPlanner)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", FakeTester)
    client = FakeClient(settings=settings)
    orch = Orchestrator(goal="g", client=client, settings=settings)
    await orch.run()
    tasks = orch._get_tasks()
    assert len(tasks) == 3
    assert all(t.status == "completed" for t in tasks)
