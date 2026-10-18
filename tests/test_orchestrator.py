from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _task(task_id: str, status: str = "completed") -> TaskModel:
    t = TaskModel(id=task_id, description=f"task {task_id}")
    t.status = status
    return t


def test_is_done_all_completed_and_tests_passed():
    orch = Orchestrator(goal="g")
    orch.tasks = [_task("1"), _task("2")]
    orch.last_test_passed = True
    assert orch._is_done() is True


def test_is_done_false_if_any_task_failed():
    orch = Orchestrator(goal="g")
    orch.tasks = [_task("1"), _task("2", status="failed")]
    orch.last_test_passed = True
    assert orch._is_done() is False


def test_is_done_false_if_tests_failed():
    orch = Orchestrator(goal="g")
    orch.tasks = [_task("1")]
    orch.last_test_passed = False
    assert orch._is_done() is False


def test_is_done_false_if_no_tasks():
    orch = Orchestrator(goal="g")
    assert orch._is_done() is False


def test_is_done_false_if_not_all_completed():
    orch = Orchestrator(goal="g")
    orch.tasks = [_task("1"), TaskModel(id="2", description="t 2")]
    orch.last_test_passed = True
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_max_cycles_halt_when_tests_keep_failing():
    fake_plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="r")
    fake_test = TestResult(passed=False, summary="fail", failures=["x"])

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
            patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
            patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        planner_instance = MockPlanner.return_value
        planner_instance.plan = AsyncMock(return_value=fake_plan)
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(return_value=fake_test)

        settings = Settings(max_cycles=1)
        orch = Orchestrator(goal="g", settings=settings)
        await orch.run()

        # The planner should run exactly once before max_cycles halts the loop.
        assert planner_instance.plan.call_count == 1
