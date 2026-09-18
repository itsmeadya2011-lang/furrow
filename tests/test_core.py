import asyncio

import pytest

from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_get_tasks_returns_merged_plans_and_is_done_tracks_state():
    orchestrator = Orchestrator("test goal", client=object())
    completed = TaskModel(id="completed", description="already done", status="completed")
    pending = TaskModel(id="pending", description="stale plan")
    orchestrator.tasks = {"completed": completed, "pending": pending}

    refreshed = TaskModel(id="pending", description="refreshed plan")
    added = TaskModel(id="added", description="new plan item")
    orchestrator._merge_tasks([refreshed, added])

    assert orchestrator._get_tasks() == [completed, refreshed, added]
    assert orchestrator._is_done() is False

    refreshed.status = "completed"
    added.status = "completed"
    assert orchestrator._is_done() is True

    orchestrator._last_test_result = TestResult(
        passed=False, summary="tests failed", failures=["failure"]
    )
    assert orchestrator._is_done() is False


def test_get_ready_tasks_waits_for_completed_dependencies():
    orchestrator = Orchestrator("test goal", client=object())
    dependency = TaskModel(id="dependency", description="run first")
    dependent = TaskModel(
        id="dependent", description="run second", dependencies=["dependency"]
    )
    orchestrator.tasks = {"dependency": dependency, "dependent": dependent}

    assert orchestrator._get_ready_tasks() == [dependency]

    dependency.status = "completed"
    assert orchestrator._get_ready_tasks() == [dependent]

    dependent.status = "completed"
    assert orchestrator._get_ready_tasks() == []


def test_mark_blocked_tasks_reports_missing_and_cyclic_dependencies():
    orchestrator = Orchestrator("test goal", client=object())
    missing_dependency = TaskModel(
        id="missing-consumer", dependencies=["not-planned"]
    )
    cycle_a = TaskModel(id="cycle-a", dependencies=["cycle-b"])
    cycle_b = TaskModel(id="cycle-b", dependencies=["cycle-a"])
    orchestrator.tasks = {
        "missing-consumer": missing_dependency,
        "cycle-a": cycle_a,
        "cycle-b": cycle_b,
    }

    orchestrator._mark_blocked_tasks()

    assert missing_dependency.status == "failed"
    assert missing_dependency.result == "missing dependencies: not-planned"
    assert cycle_a.status == cycle_b.status == "failed"
    assert cycle_a.result is not None
    assert cycle_b.result is not None
    assert "cyclic or unresolvable dependencies" in cycle_a.result
    assert "cycle-b" in cycle_a.result
    assert "cyclic or unresolvable dependencies" in cycle_b.result
    assert "cycle-a" in cycle_b.result


async def _assert_worker_concurrency(
    monkeypatch, max_parallel_tasks: int, task_count: int
) -> None:
    active_workers = 0
    max_active_workers = 0

    class FakeWorkerAgent:
        def __init__(self, task: TaskModel, client: object | None = None) -> None:
            self.task = task

        async def run(self) -> str:
            nonlocal active_workers, max_active_workers
            active_workers += 1
            max_active_workers = max(max_active_workers, active_workers)
            try:
                await asyncio.sleep(0.01)
                return self.task.id
            finally:
                active_workers -= 1

    monkeypatch.setattr(settings, "max_parallel_tasks", max_parallel_tasks)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorkerAgent)
    orchestrator = Orchestrator("test goal", client=object())
    tasks = [TaskModel(id=str(index), description=f"task {index}") for index in range(task_count)]

    results = await orchestrator._execute_tasks(tasks)

    assert results == [task.id for task in tasks]
    assert active_workers == 0
    assert max_active_workers == (
        task_count if max_parallel_tasks == 0 else max_parallel_tasks
    )


@pytest.mark.asyncio
async def test_execute_tasks_respects_positive_parallel_cap(monkeypatch):
    await _assert_worker_concurrency(monkeypatch, max_parallel_tasks=2, task_count=5)


@pytest.mark.asyncio
async def test_execute_tasks_treats_zero_as_unlimited(monkeypatch):
    await _assert_worker_concurrency(monkeypatch, max_parallel_tasks=0, task_count=5)


@pytest.mark.asyncio
async def test_run_stops_at_configured_max_cycles_with_fakes(monkeypatch):
    monkeypatch.setattr(settings, "max_cycles", 2)
    plan = Plan(tasks=[TaskModel(id="task", description="run")], rationale="test")

    class FakePlannerAgent:
        def __init__(self, client: object | None = None) -> None:
            pass

        async def plan(self, goal: str) -> Plan:
            return plan

    class FakeWorkerAgent:
        def __init__(self, task: TaskModel, client: object | None = None) -> None:
            pass

        async def run(self) -> str:
            return "done"

    class FakeTesterAgent:
        def __init__(self, client: object | None = None) -> None:
            pass

        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            return TestResult(
                passed=False, summary="keep iterating", failures=["keep iterating"]
            )

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", FakePlannerAgent)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorkerAgent)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", FakeTesterAgent)
    orchestrator = Orchestrator("test goal", client=object())

    await orchestrator.run()

    assert orchestrator.cycles == settings.max_cycles


@pytest.mark.asyncio
async def test_run_halts_when_failed_tests_are_followed_by_an_empty_plan(
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "max_cycles", 3)
    planner_calls = 0
    tester_calls = 0

    plan = Plan(tasks=[TaskModel(id="task", description="run")], rationale="test")
    empty_plan = Plan(tasks=[], rationale="no more work")
    failed_result = TestResult(
        passed=False,
        summary="tests failed",
        failures=["failure"],
    )

    class FakePlannerAgent:
        def __init__(self, client: object | None = None) -> None:
            pass

        async def plan(self, goal: str) -> Plan:
            nonlocal planner_calls
            planner_calls += 1
            return plan if planner_calls == 1 else empty_plan

    class FakeWorkerAgent:
        def __init__(self, task: TaskModel, client: object | None = None) -> None:
            pass

        async def run(self) -> str:
            return "done"

    class FakeTesterAgent:
        def __init__(self, client: object | None = None) -> None:
            pass

        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            nonlocal tester_calls
            tester_calls += 1
            return failed_result

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", FakePlannerAgent)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorkerAgent)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", FakeTesterAgent)
    orchestrator = Orchestrator("test goal", client=object())

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert planner_calls == 2
    assert tester_calls == 1
    assert orchestrator._last_test_result is failed_result
    assert orchestrator._last_test_result.passed is False
