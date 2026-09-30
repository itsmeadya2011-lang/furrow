import asyncio
from unittest.mock import AsyncMock, Mock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def _make_mock_client():
    client = Mock(spec=LLMClient)
    client.settings = Mock()
    client.settings.max_cycles = 0
    client.settings.max_parallel_tasks = 5
    return client


def test_orchestrator_is_done_empty_tasks():
    client = _make_mock_client()
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.tasks = []
    orchestrator.last_test_result = None
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_all_completed_tests_pass():
    client = _make_mock_client()
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.tasks = [TaskModel(id="1", description="task", status="completed")]
    orchestrator.last_test_result = TestResult(passed=True, summary="all good", failures=[])
    assert orchestrator._is_done() is True


def test_orchestrator_is_done_task_failed():
    client = _make_mock_client()
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.tasks = [TaskModel(id="1", description="task", status="failed")]
    orchestrator.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orchestrator._is_done() is False


def test_orchestrator_is_done_tests_failed():
    client = _make_mock_client()
    orchestrator = Orchestrator(goal="test", client=client)
    orchestrator.tasks = [TaskModel(id="1", description="task", status="completed")]
    orchestrator.last_test_result = TestResult(passed=False, summary="fail", failures=["test failed"])
    assert orchestrator._is_done() is False


async def test_orchestrator_stores_tasks():
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing", status="pending")],
        rationale="ok",
    )
    client = _make_mock_client()

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as MockWorker, patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=True, summary="ok", failures=[])
        )

        orchestrator = Orchestrator(goal="test", client=client)
        await orchestrator._cycle()

        assert orchestrator.tasks == plan.tasks
        assert orchestrator.tasks[0].status == "completed"


async def test_orchestrator_max_cycles():
    plan = Plan(
        tasks=[TaskModel(id="1", description="task", status="pending")],
        rationale="ok",
    )
    client = _make_mock_client()
    client.settings.max_cycles = 1

    failing_test = TestResult(passed=False, summary="failure", failures=["broken"])

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as MockWorker, patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(return_value=failing_test)

        orchestrator = Orchestrator(goal="test", client=client)
        await orchestrator.run()

        assert orchestrator.cycles == 1


async def test_orchestrator_max_parallel_tasks():
    plan = Plan(
        tasks=[TaskModel(id=str(i), description=f"task {i}", status="pending") for i in range(5)],
        rationale="ok",
    )
    client = _make_mock_client()
    client.settings.max_parallel_tasks = 2

    current = 0
    peak = 0

    async def track_run(*args, **kwargs):
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return "done"

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, patch(
        "furrow.core.orchestrator.WorkerAgent"
    ) as MockWorker, patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockPlanner.return_value.plan = AsyncMock(return_value=plan)
        MockWorker.return_value.run = AsyncMock(side_effect=track_run)
        MockTester.return_value.run = AsyncMock(
            return_value=TestResult(passed=False, summary="fail", failures=["x"])
        )

        orchestrator = Orchestrator(goal="test", client=client)
        await orchestrator._cycle()

        assert peak <= 2
        assert peak == 2
