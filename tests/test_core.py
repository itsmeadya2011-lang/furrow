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
    def __init__(self, plan: Plan) -> None:
        self.plan = plan

    async def plan(self, goal: str) -> Plan:
        return self.plan


class PassingWorkerAgent:
    def __init__(self, task: TaskModel, client: FakeClient) -> None:
        self.task = task

    async def run(self) -> str:
        return f"completed {self.task.id}"


class PassingTesterAgent:
    def __init__(self, client: FakeClient) -> None:
        pass

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])


class FailingWorkerAgent:
    def __init__(self, task: TaskModel, client: FakeClient) -> None:
        pass

    async def run(self) -> str:
        raise RuntimeError("still failing")


class FailingTesterAgent:
    def __init__(self, client: FakeClient) -> None:
        pass

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return TestResult(passed=False, summary="failed", failures=["still failing"])


class CompletedWorkerAgent:
    """Worker that completes its task (returns a result string)."""

    def __init__(self, task: TaskModel, client: FakeClient) -> None:
        self.task = task

    async def run(self) -> str:
        return f"completed {self.task.id}"


def test_plan_parse() -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert plan.tasks[0].description == "do thing"


def test_test_result() -> None:
    result = TestResult(passed=True, summary="ok", failures=[])
    assert result.passed is True


@pytest.mark.asyncio
async def test_completed_tasks_with_failing_tests_do_not_halt_before_max_cycles() -> None:
    task = TaskModel(id="task-1", description="implement feature")
    plan = Plan(tasks=[task], rationale="one task")
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=3))
    orchestrator.planner = FakePlanner(plan)

    with (
        patch("furrow.core.orchestrator.WorkerAgent", CompletedWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", FailingTesterAgent),
    ):
        await orchestrator.run()

    # Tasks completed but tests never passed, so the loop only stops at max_cycles.
    assert orchestrator.cycles == 3
    assert task.status == "completed"
    assert orchestrator.tests_passed is False
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_passing_tests_complete_after_all_tasks_done() -> None:
    task = TaskModel(id="task-1", description="implement feature")
    plan = Plan(tasks=[task], rationale="one task")
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=0))
    orchestrator.planner = FakePlanner(plan)

    with (
        patch("furrow.core.orchestrator.WorkerAgent", CompletedWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", PassingTesterAgent),
    ):
        await orchestrator.run()

    assert orchestrator.cycles == 1
    assert task.status == "completed"
    assert orchestrator.tests_passed is True
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_cycle_retains_completed_task_state() -> None:
    task = TaskModel(id="task-1", description="implement feature")
    plan = Plan(tasks=[task], rationale="one task")
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=0))
    orchestrator.planner = FakePlanner(plan)

    assert orchestrator._get_tasks() == []

    with (
        patch("furrow.core.orchestrator.WorkerAgent", PassingWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", PassingTesterAgent),
    ):
        await orchestrator._cycle()

    assert orchestrator._get_tasks() == [task]
    assert task.status == "completed"
    assert task.result == "completed task-1"
    assert orchestrator._is_done() is True


class CountingPlanner:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        return self.plan


class SwitchingPlanner:
    """Planner that returns one plan for the first call, another thereafter."""

    def __init__(self, first_plan: Plan, second_plan: Plan) -> None:
        self.first_plan = first_plan
        self.second_plan = second_plan
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        if self.calls == 1:
            return self.first_plan
        return self.second_plan


@pytest.mark.asyncio
async def test_run_stops_after_positive_max_cycles() -> None:
    task = TaskModel(id="task-1", description="implement feature")
    plan = Plan(tasks=[task], rationale="one task")
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=2))
    planner = CountingPlanner(plan)
    orchestrator.planner = planner

    with (
        patch("furrow.core.orchestrator.WorkerAgent", FailingWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", FailingTesterAgent),
    ):
        await orchestrator.run()

    assert orchestrator.cycles == 2
    assert planner.calls == 2
    assert task.status == "failed"


class PlannerExhausted(Exception):
    pass


class ExhaustingPlanner:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        if self.calls > 3:
            raise PlannerExhausted
        return self.plan


@pytest.mark.asyncio
async def test_run_with_zero_max_cycles_is_unlimited() -> None:
    task = TaskModel(id="task-1", description="implement feature")
    plan = Plan(tasks=[task], rationale="one task")
    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=0))
    planner = ExhaustingPlanner(plan)
    orchestrator.planner = planner

    with (
        patch("furrow.core.orchestrator.WorkerAgent", FailingWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", FailingTesterAgent),
    ):
        with pytest.raises(PlannerExhausted):
            await orchestrator.run()

    assert orchestrator.cycles == 3
    assert planner.calls == 4


@pytest.mark.asyncio
async def test_empty_plan_after_failing_cycle_completes() -> None:
    """A no-task plan following a cycle whose tasks/tests failed must still be
    recognized as complete, clearing stale task/test state."""
    task = TaskModel(id="task-1", description="implement feature")
    failing_plan = Plan(tasks=[task], rationale="one task")
    empty_plan = Plan(tasks=[], rationale="goal complete")
    planner = SwitchingPlanner(failing_plan, empty_plan)

    orchestrator = Orchestrator("build feature", client=FakeClient(max_cycles=5))
    orchestrator.planner = planner

    with (
        patch("furrow.core.orchestrator.WorkerAgent", FailingWorkerAgent),
        patch("furrow.core.orchestrator.TesterAgent", FailingTesterAgent),
    ):
        await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator._get_tasks() == []
    assert orchestrator.tests_passed is None
    assert orchestrator._is_done() is True
    assert planner.calls == 2
