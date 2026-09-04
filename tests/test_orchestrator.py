from unittest.mock import AsyncMock, patch

import pytest
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_is_done_no_tasks():
    orch = Orchestrator(goal="test")
    orch._current_tasks = []
    assert orch._is_done() is True


def test_is_done_all_completed_tests_passed():
    orch = Orchestrator(goal="test")
    orch._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch.last_test_passed = True
    assert orch._is_done() is True


def test_is_done_task_failed():
    orch = Orchestrator(goal="test")
    orch._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    orch.last_test_passed = True
    assert orch._is_done() is False


def test_is_done_task_pending():
    orch = Orchestrator(goal="test")
    orch._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    orch.last_test_passed = True
    assert orch._is_done() is False


def test_is_done_tests_failed():
    orch = Orchestrator(goal="test")
    orch._current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
    ]
    orch.last_test_passed = False
    assert orch._is_done() is False


def test_get_tasks_returns_current_tasks():
    orch = Orchestrator(goal="test")
    tasks = [TaskModel(id="1", description="a")]
    orch._current_tasks = tasks
    assert orch._get_tasks() is tasks


def test_settings_optional():
    orch = Orchestrator(goal="test")
    assert orch.settings is not None


@pytest.mark.asyncio
async def test_max_cycles_breaks_loop():
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    test_result = TestResult(passed=False, summary="fail", failures=["x"])

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester, \
         patch("furrow.core.orchestrator.LLMClient"):
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(return_value=test_result)

        settings = Settings(max_cycles=2)
        orch = Orchestrator(goal="test", settings=settings)
        await orch.run()

        assert orch.cycles == 2


@pytest.mark.asyncio
async def test_all_tasks_accumulates():
    plan1 = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    plan2 = Plan(tasks=[TaskModel(id="2", description="b")], rationale="ok")
    test_result_fail = TestResult(passed=False, summary="fail", failures=["x"])
    test_result_pass = TestResult(passed=True, summary="ok", failures=[])

    results = [
        (plan1, test_result_fail),
        (plan2, test_result_pass),
    ]
    result_iter = iter(results)

    async def mock_plan(*args, **kwargs):
        return next(result_iter)[0]

    async def mock_tester_run(*args, **kwargs):
        return next(result_iter)[1]

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester, \
         patch("furrow.core.orchestrator.LLMClient"):
        MockPlanner.return_value.plan = mock_plan
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = mock_tester_run

        settings = Settings(max_cycles=0)
        orch = Orchestrator(goal="test", settings=settings)
        await orch.run()

        assert len(orch.all_tasks) == 2
        assert orch.all_tasks[0].id == "1"
        assert orch.all_tasks[1].id == "2"


@pytest.mark.asyncio
async def test_current_tasks_set_per_cycle():
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="ok")
    test_result = TestResult(passed=True, summary="ok", failures=[])

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester, \
         patch("furrow.core.orchestrator.LLMClient"):
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(return_value=test_result)

        settings = Settings(max_cycles=0)
        orch = Orchestrator(goal="test", settings=settings)
        await orch.run()

        assert len(orch._current_tasks) == 1
        assert orch._current_tasks[0].id == "1"
