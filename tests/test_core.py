from functools import partial
from unittest.mock import patch

import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakeSettings:
    def __init__(self, max_cycles: int) -> None:
        self.max_cycles = max_cycles


class FakeClient:
    def __init__(self, max_cycles: int) -> None:
        self.settings = FakeSettings(max_cycles)


class FakePlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = plans
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        plan = self.plans[self.calls]
        self.calls += 1
        return plan


class FakeWorkerAgent:
    def __init__(self, task: TaskModel, client: FakeClient) -> None:
        self.task = task

    async def run(self) -> str:
        return f"result:{self.task.id}"


class FakeTesterAgent:
    def __init__(
        self,
        client: FakeClient,
        results: list[TestResult] | None = None,
    ) -> None:
        self.results = results or [TestResult(passed=True, summary="ok", failures=[])]
        self.calls = 0

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        result = self.results[self.calls]
        self.calls += 1
        return result


class FailingWorkerAgent:
    def __init__(self, task: TaskModel, client: FakeClient) -> None:
        self.task = task

    async def run(self) -> str:
        raise RuntimeError("worker failed")


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [
        ([], True),
        (["pending"], False),
        (["completed"], True),
        (["failed"], True),
        (["completed", "pending"], False),
        (["completed", "failed"], True),
    ],
)
def test_is_done_evaluates_current_task_statuses(statuses: list[str], expected: bool) -> None:
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.tasks = [
        TaskModel(id=str(index), description="task", status=status)
        for index, status in enumerate(statuses)
    ]

    assert orchestrator._is_done() is expected


@pytest.mark.asyncio
async def test_cycle_retains_and_resets_planned_tasks() -> None:
    first_task = TaskModel(id="first", description="do first thing")
    second_task = TaskModel(id="second", description="do second thing")
    planner = FakePlanner(
        [
            Plan(tasks=[first_task], rationale="first plan"),
            Plan(tasks=[second_task], rationale="second plan"),
        ]
    )
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=0))
    orchestrator.planner = planner

    with (
        patch("furrow.core.orchestrator.WorkerAgent", FakeWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", FakeTesterAgent),
    ):
        await orchestrator._cycle()
        assert orchestrator._get_tasks() == [first_task]
        assert first_task.status == "completed"
        assert first_task.result == "result:first"

        assert second_task.status == "pending"
        await orchestrator._cycle()

    assert orchestrator._get_tasks() == [second_task]
    assert first_task.status == "completed"
    assert first_task.result == "result:first"
    assert second_task.status == "completed"
    assert second_task.result == "result:second"
    assert planner.calls == 2


@pytest.mark.asyncio
async def test_run_continues_after_test_failure_until_tests_pass() -> None:
    first_task = TaskModel(id="first", description="do first thing")
    second_task = TaskModel(id="second", description="do second thing")
    planner = FakePlanner(
        [
            Plan(tasks=[first_task], rationale="first plan"),
            Plan(tasks=[second_task], rationale="second plan"),
        ]
    )
    failing_result = TestResult(
        passed=False,
        summary="failure",
        failures=["test failed"],
    )
    passing_result = TestResult(passed=True, summary="ok", failures=[])
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=2))
    orchestrator.planner = planner
    tester_factory = partial(
        FakeTesterAgent,
        results=[failing_result, passing_result],
    )

    with (
        patch("furrow.core.orchestrator.WorkerAgent", FakeWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", tester_factory),
    ):
        await orchestrator.run()

    assert orchestrator.cycles == 2
    assert planner.calls == 2
    assert first_task.status == "completed"
    assert second_task.status == "completed"
    assert orchestrator.goal == "Fix failing tests:\ntest failed"
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_run_stops_at_positive_max_cycles_without_llm_calls() -> None:
    task = TaskModel(id="task-1", description="do thing")
    plan = Plan(tasks=[task], rationale="one task")
    planner = FakePlanner([plan])
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=2))
    orchestrator.planner = planner
    orchestrator._is_done = lambda: False

    with (
        patch("furrow.core.orchestrator.WorkerAgent", FakeWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", FakeTesterAgent),
    ):
        await orchestrator.run()

    assert orchestrator.cycles == 2
    assert planner.calls == 2
    assert task.status == "completed"
    assert task.result == "result:task-1"


@pytest.mark.asyncio
async def test_run_terminates_after_empty_plan() -> None:
    planner = FakePlanner([Plan(tasks=[], rationale="goal complete")])
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=0))
    orchestrator.planner = planner

    with (
        patch("furrow.core.orchestrator.WorkerAgent") as worker_factory,
        patch("furrow.core.orchestrator.TesterAgent") as tester_factory,
    ):
        await orchestrator.run()

    assert orchestrator.cycles == 1
    assert planner.calls == 1
    assert orchestrator._get_tasks() == []
    assert orchestrator._is_done() is True
    worker_factory.assert_not_called()
    tester_factory.assert_not_called()


@pytest.mark.asyncio
async def test_run_stops_after_worker_failure() -> None:
    task = TaskModel(id="task-1", description="do thing")
    next_task = TaskModel(id="task-2", description="do next thing")
    planner = FakePlanner(
        [
            Plan(tasks=[task], rationale="first plan"),
            Plan(tasks=[next_task], rationale="second plan"),
        ]
    )
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=2))
    orchestrator.planner = planner

    with (
        patch("furrow.core.orchestrator.WorkerAgent", FailingWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", FakeTesterAgent),
    ):
        await orchestrator.run()

    assert orchestrator.cycles == 1
    assert planner.calls == 1
    assert task.status == "failed"
    assert task.result == "worker failed"
    assert next_task.status == "pending"
    assert orchestrator._is_done() is True
