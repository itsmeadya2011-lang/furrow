from __future__ import annotations

import asyncio

import pytest

from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.orchestrator import CycleLimitError, DeadlockError, Orchestrator


class FakePlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = plans
        self.goals: list[str] = []
        self._index = 0

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        plan = self.plans[min(self._index, len(self.plans) - 1)]
        self._index += 1
        return plan


class FakeTester:
    def __init__(self, results: list[TestResult]) -> None:
        self.results = results
        self.goals: list[str] = []
        self.task_ids: list[list[str]] = []
        self._index = 0

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.goals.append(goal)
        self.task_ids.append([task.id for task in tasks])
        result = self.results[min(self._index, len(self.results) - 1)]
        self._index += 1
        return result


def _settings(**overrides: int) -> Settings:
    return Settings(max_parallel_tasks=5, max_cycles=10, **overrides)


def _task(task_id: str, dependencies: list[str] | None = None) -> TaskModel:
    return TaskModel(
        id=task_id,
        description=f"complete {task_id}",
        dependencies=dependencies or [],
    )


def _plan(*tasks: TaskModel) -> Plan:
    return Plan(tasks=list(tasks), rationale="test plan")


@pytest.mark.asyncio
async def test_dependency_ordering() -> None:
    plan = _plan(
        _task("first"),
        _task("second", ["first"]),
        _task("third", ["second"]),
    )
    planner = FakePlanner([plan])
    calls: list[str] = []

    async def worker_factory(task: TaskModel) -> str:
        calls.append(task.id)
        return f"done {task.id}"

    tester = FakeTester([TestResult(passed=True, summary="ok")])
    orchestrator = Orchestrator(
        goal="build in order",
        settings=_settings(max_cycles=3),
        planner=planner,
        worker_factory=worker_factory,
        tester=tester,
    )

    await orchestrator.run()

    assert calls == ["first", "second", "third"]
    assert [orchestrator.tasks[task_id].status for task_id in calls] == [
        TaskStatus.COMPLETED,
        TaskStatus.COMPLETED,
        TaskStatus.COMPLETED,
    ]


@pytest.mark.asyncio
async def test_max_parallel_tasks_limits_concurrency() -> None:
    plan = _plan(*[_task(f"task-{index}") for index in range(4)])
    planner = FakePlanner([plan])
    calls: list[str] = []
    active = 0
    max_active = 0

    async def worker_factory(task: TaskModel) -> str:
        nonlocal active, max_active
        calls.append(task.id)
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return task.id

    tester = FakeTester([TestResult(passed=True, summary="ok")])
    orchestrator = Orchestrator(
        goal="run parallel tasks",
        settings=_settings(max_parallel_tasks=2, max_cycles=2),
        planner=planner,
        worker_factory=worker_factory,
        tester=tester,
    )

    await orchestrator.run()

    assert len(calls) == 4
    assert max_active == 2
    assert active == 0


@pytest.mark.asyncio
async def test_failed_dependency_is_blocked_and_deadlocks() -> None:
    planner = FakePlanner(
        [
            _plan(_task("build"), _task("deploy", ["build"])),
            Plan(tasks=[], rationale="no more work"),
        ]
    )
    calls: list[str] = []

    async def worker_factory(task: TaskModel) -> str:
        calls.append(task.id)
        raise RuntimeError("build failed")

    tester = FakeTester([TestResult(passed=True, summary="ok")])
    orchestrator = Orchestrator(
        goal="deploy after build",
        settings=_settings(max_cycles=3),
        planner=planner,
        worker_factory=worker_factory,
        tester=tester,
    )

    with pytest.raises(DeadlockError, match="No ready tasks"):
        await orchestrator.run()

    assert calls == ["build"]
    blocked = orchestrator.tasks["deploy"]
    assert blocked.status is TaskStatus.BLOCKED
    assert "build" in (blocked.error or "")


@pytest.mark.asyncio
async def test_cycle_limit_after_repeated_worker_failure() -> None:
    plan = _plan(_task("unstable"))
    planner = FakePlanner([plan])
    attempts = 0

    async def worker_factory(task: TaskModel) -> str:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("worker failed")

    tester = FakeTester([TestResult(passed=True, summary="ok")])
    orchestrator = Orchestrator(
        goal="retry unstable work",
        settings=_settings(max_cycles=2),
        planner=planner,
        worker_factory=worker_factory,
        tester=tester,
    )

    with pytest.raises(CycleLimitError, match="max cycles"):
        await orchestrator.run()

    assert attempts == 2
    assert orchestrator.cycles == 2
    assert orchestrator.tasks["unstable"].status is TaskStatus.FAILED


@pytest.mark.asyncio
async def test_repair_after_test_failure_preserves_original_goal() -> None:
    goal = "make the feature work"
    plan = _plan(_task("feature"))
    planner = FakePlanner([plan])
    attempts = 0

    async def worker_factory(task: TaskModel) -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("broken implementation")
        return "fixed implementation"

    tester = FakeTester(
        [
            TestResult(
                passed=False,
                summary="one failure",
                failures=["missing feature"],
            ),
            TestResult(passed=True, summary="all tests pass"),
        ]
    )
    orchestrator = Orchestrator(
        goal=goal,
        settings=_settings(max_cycles=3),
        planner=planner,
        worker_factory=worker_factory,
        tester=tester,
    )

    await orchestrator.run()

    assert orchestrator.original_goal == goal
    assert orchestrator.current_goal.startswith(f"Original goal: {goal}")
    assert "missing feature" in orchestrator.current_goal
    assert planner.goals[0] == goal
    assert f"Original goal: {goal}" in planner.goals[1]
    assert "missing feature" in planner.goals[1]
    assert tester.goals == [goal, goal]
    assert attempts == 2
    assert orchestrator.tasks["feature"].result == "fixed implementation"
