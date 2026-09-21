import asyncio

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator


class FakePlanner:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.goals = []
        self.calls = 0

    async def plan(self, goal):
        self.calls += 1
        self.goals.append(goal)
        return next(self.plans)


class FakeTester:
    def __init__(self, results, client=None, settings=None):
        self.results = iter(results)
        self.calls = 0

    async def run(self, goal, tasks):
        self.calls += 1
        return next(self.results)


def make_settings(**overrides):
    values = {
        "max_cycles": 0,
        "max_parallel_tasks": 2,
        "workspace": ".",
        "test_timeout": 1.0,
    }
    values.update(overrides)
    return Settings(**values)


def task(task_id="1"):
    return TaskModel(id=task_id, description=f"task {task_id}")


@pytest.mark.asyncio
async def test_failed_verification_cannot_complete_and_preserves_goal(monkeypatch):
    settings = make_settings()
    orchestrator = Orchestrator("original goal", settings=settings)
    orchestrator.planner = FakePlanner(
        [Plan(tasks=[task()], rationale="implement")]
    )
    tester = FakeTester([TestResult(passed=False, summary="failed", failures=["boom"])])
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda **kwargs: tester)

    async def execute(tasks):
        return ["done" for _ in tasks]

    monkeypatch.setattr(orchestrator, "_execute_tasks", execute)

    await orchestrator._cycle()

    assert orchestrator.tasks[0].status == "completed"
    assert orchestrator.latest_test_result.passed is False
    assert orchestrator._is_done() is False
    assert "original goal" in orchestrator.goal
    assert "boom" in orchestrator.goal


@pytest.mark.asyncio
async def test_passing_verification_can_complete(monkeypatch):
    settings = make_settings()
    orchestrator = Orchestrator("original goal", settings=settings)
    orchestrator.planner = FakePlanner(
        [Plan(tasks=[task()], rationale="implement")]
    )
    tester = FakeTester([TestResult(passed=True, summary="passed", failures=[])])
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda **kwargs: tester)

    async def execute(tasks):
        return ["done" for _ in tasks]

    monkeypatch.setattr(orchestrator, "_execute_tasks", execute)

    await orchestrator._cycle()

    assert orchestrator.tasks[0].status == "completed"
    assert orchestrator.latest_test_result.passed is True
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_empty_plan_does_not_false_success(monkeypatch):
    settings = make_settings()
    orchestrator = Orchestrator("original goal", settings=settings)
    orchestrator.planner = FakePlanner([Plan(tasks=[], rationale="nothing")])
    tester = FakeTester([TestResult(passed=True, summary="passed", failures=[])])
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda **kwargs: tester)

    await orchestrator._cycle()

    assert orchestrator.tasks == []
    assert orchestrator.latest_test_result.passed is True
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_task_state_is_retained_across_plans(monkeypatch):
    settings = make_settings()
    orchestrator = Orchestrator("original goal", settings=settings)
    orchestrator.planner = FakePlanner(
        [
            Plan(tasks=[task("1")], rationale="first"),
            Plan(tasks=[task("2")], rationale="second"),
        ]
    )
    tester = FakeTester(
        [
            TestResult(passed=False, summary="first failed", failures=["one"]),
            TestResult(passed=True, summary="passed", failures=[]),
        ]
    )
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda **kwargs: tester)

    async def execute(tasks):
        return ["done" for _ in tasks]

    monkeypatch.setattr(orchestrator, "_execute_tasks", execute)

    await orchestrator._cycle()
    await orchestrator._cycle()

    assert [item.id for item in orchestrator.tasks] == ["1", "2"]
    assert [item.status for item in orchestrator.tasks] == ["completed", "completed"]
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_max_cycles_stops_after_configured_count(monkeypatch):
    settings = make_settings(max_cycles=2)
    orchestrator = Orchestrator("goal", settings=settings)
    planner = FakePlanner(
        [Plan(tasks=[task("1")], rationale="one"), Plan(tasks=[task("2")], rationale="two")]
    )
    orchestrator.planner = planner
    tester = FakeTester(
        [
            TestResult(passed=False, summary="failed", failures=["one"]),
            TestResult(passed=False, summary="failed", failures=["two"]),
        ]
    )
    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda **kwargs: tester)

    async def execute(tasks):
        return ["done" for _ in tasks]

    monkeypatch.setattr(orchestrator, "_execute_tasks", execute)

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert planner.calls == 2
    assert len(orchestrator.tasks) == 2


@pytest.mark.asyncio
async def test_max_parallel_tasks_limits_worker_concurrency(monkeypatch):
    settings = make_settings(max_parallel_tasks=1)
    orchestrator = Orchestrator("goal", settings=settings)
    active = 0
    max_active = 0

    class FakeWorker:
        def __init__(self, task, client=None, settings=None):
            self.task = task

        async def run(self):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            active -= 1
            return "done"

    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)

    results = await orchestrator._execute_tasks([task("1"), task("2"), task("3")])

    assert results == ["done", "done", "done"]
    assert max_active == 1
