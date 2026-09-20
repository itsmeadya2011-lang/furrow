import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator

GOAL = "Implement the feature"


@pytest.mark.asyncio
async def test_default_construction_uses_settings() -> None:
    orchestrator = Orchestrator(GOAL)

    settings = orchestrator.settings
    client = orchestrator.client
    assert settings is not None
    assert settings is client.settings
    assert isinstance(orchestrator.max_cycles, int)


@pytest.mark.asyncio
async def test_worker_cancellation_propagates() -> None:
    task = TaskModel(id="task-1", description="Implement feature")
    plan = Plan(tasks=[task], rationale="One focused task")

    with (
        patch("furrow.core.orchestrator.PlannerAgent") as planner_cls,
        patch("furrow.core.orchestrator.WorkerAgent") as worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as tester_cls,
    ):
        planner_cls.return_value.plan = AsyncMock(return_value=plan)
        worker_cls.return_value.run = AsyncMock(side_effect=asyncio.CancelledError())
        orchestrator = Orchestrator(GOAL, client=object())

        with pytest.raises(asyncio.CancelledError):
            await orchestrator.run()

    assert orchestrator.cycles == 1
    assert task.status == "pending"
    tester_cls.assert_not_called()


def settings(max_cycles: int = 0) -> Settings:
    return Settings(max_cycles=max_cycles, _env_file=None)


@pytest.mark.asyncio
async def test_passing_result_completes_after_one_cycle() -> None:
    task = TaskModel(id="task-1", description="Implement feature")
    plan = Plan(tasks=[task], rationale="One focused task")

    with (
        patch("furrow.core.orchestrator.PlannerAgent") as planner_cls,
        patch("furrow.core.orchestrator.WorkerAgent") as worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as tester_cls,
    ):
        planner_cls.return_value.plan = AsyncMock(return_value=plan)
        worker_cls.return_value.run = AsyncMock(return_value="complete")
        tester_cls.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all checks passed")
        )
        orchestrator = Orchestrator(GOAL, client=object())

        await orchestrator.run()

    assert orchestrator.cycles == 1
    assert orchestrator.completed
    assert orchestrator.latest_test_result is not None
    assert orchestrator.latest_test_result.passed
    assert orchestrator._get_tasks() == [task]


@pytest.mark.asyncio
async def test_failed_then_passing_result_repairs_goal_and_runs_twice() -> None:
    task = TaskModel(id="task-1", description="Implement feature")
    plan = Plan(tasks=[task], rationale="Repair the implementation")
    failed_test = TestResult(
        passed=False,
        summary="checks failed",
        failures=["test_feature failed"],
    )
    passing_test = TestResult(passed=True, summary="all checks passed")

    with (
        patch("furrow.core.orchestrator.PlannerAgent") as planner_cls,
        patch("furrow.core.orchestrator.WorkerAgent") as worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as tester_cls,
    ):
        planner_cls.return_value.plan = AsyncMock(side_effect=[plan, plan])
        worker_cls.return_value.run = AsyncMock(return_value="complete")
        tester_cls.return_value.run = AsyncMock(side_effect=[failed_test, passing_test])
        orchestrator = Orchestrator(GOAL, client=object())

        await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator.completed
    assert planner_cls.return_value.plan.await_count == 2
    repair_goal = planner_cls.return_value.plan.await_args_list[1].args[0]
    assert GOAL in repair_goal
    assert "Fix failing tests" in repair_goal


@pytest.mark.asyncio
async def test_worker_exception_marks_task_failed_and_halts_safely() -> None:
    task = TaskModel(id="task-1", description="Implement feature")
    plan = Plan(tasks=[task], rationale="One focused task")

    with (
        patch("furrow.core.orchestrator.PlannerAgent") as planner_cls,
        patch("furrow.core.orchestrator.WorkerAgent") as worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as tester_cls,
    ):
        planner_cls.return_value.plan = AsyncMock(return_value=plan)
        worker_cls.return_value.run = AsyncMock(side_effect=RuntimeError("worker failed"))
        tester_cls.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="all checks passed")
        )
        orchestrator = Orchestrator(
            GOAL,
            client=object(),
            settings=settings(max_cycles=1),
        )

        await orchestrator.run()

    assert task.status == "failed"
    assert task.result == "worker failed"
    assert not orchestrator.completed
    assert orchestrator.halt_reason == "Verification passed, but one or more tasks failed."


@pytest.mark.asyncio
async def test_max_cycles_stops_after_repeated_failures() -> None:
    task = TaskModel(id="task-1", description="Implement feature")
    plan = Plan(tasks=[task], rationale="Retry the implementation")
    failed_test = TestResult(
        passed=False,
        summary="checks failed",
        failures=["test_feature failed"],
    )

    with (
        patch("furrow.core.orchestrator.PlannerAgent") as planner_cls,
        patch("furrow.core.orchestrator.WorkerAgent") as worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as tester_cls,
    ):
        planner_cls.return_value.plan = AsyncMock(side_effect=[plan, plan])
        worker_cls.return_value.run = AsyncMock(return_value="complete")
        tester_cls.return_value.run = AsyncMock(side_effect=[failed_test, failed_test])
        orchestrator = Orchestrator(
            GOAL,
            client=object(),
            settings=settings(max_cycles=2),
        )

        await orchestrator.run()

    assert orchestrator.cycles == 2
    assert not orchestrator.completed
    assert orchestrator.halt_reason == "Maximum cycle limit of 2 reached."
    assert planner_cls.return_value.plan.await_count == 2


@pytest.mark.asyncio
async def test_initial_empty_plan_completes_safely() -> None:
    with (
        patch("furrow.core.orchestrator.PlannerAgent") as planner_cls,
        patch("furrow.core.orchestrator.WorkerAgent") as worker_cls,
        patch("furrow.core.orchestrator.TesterAgent") as tester_cls,
    ):
        planner_cls.return_value.plan = AsyncMock(
            return_value=Plan(tasks=[], rationale="No work required")
        )
        orchestrator = Orchestrator(GOAL, client=object())

        await orchestrator.run()

    assert orchestrator.cycles == 1
    assert orchestrator.completed
    assert orchestrator._is_done()
    assert orchestrator.halt_reason is None
    assert orchestrator._get_tasks() == []
    worker_cls.assert_not_called()
    tester_cls.assert_not_called()
