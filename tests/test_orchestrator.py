import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.settings.max_cycles = 2
    client.settings.max_parallel_tasks = 3
    return client


@pytest.fixture
def orchestrator(mock_client):
    return Orchestrator(goal="test goal", client=mock_client)


def test_orchestrator_init(orchestrator):
    assert orchestrator.goal == "test goal"
    assert orchestrator.original_goal == "test goal"
    assert orchestrator.current_plan is None
    assert orchestrator.cycles == 0


def test_get_tasks_empty(orchestrator):
    assert orchestrator._get_tasks() == []


def test_get_tasks_with_plan(orchestrator):
    tasks = [TaskModel(id="t1", description="task one")]
    orchestrator.current_plan = Plan(tasks=tasks, rationale="r")
    assert orchestrator._get_tasks() == tasks


def test_is_done_with_failed_tasks(orchestrator):
    tasks = [
        TaskModel(id="t1", description="task one", status="completed"),
        TaskModel(id="t2", description="task two", status="failed"),
    ]
    orchestrator.current_plan = Plan(tasks=tasks, rationale="r")
    assert orchestrator._is_done() is False


def test_is_done_all_completed(orchestrator):
    tasks = [
        TaskModel(id="t1", description="task one", status="completed"),
        TaskModel(id="t2", description="task two", status="completed"),
    ]
    orchestrator.current_plan = Plan(tasks=tasks, rationale="r")
    assert orchestrator._is_done() is True


def test_is_done_no_tasks(orchestrator):
    orchestrator.current_plan = Plan(tasks=[], rationale="r")
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_max_cycles_enforcement(mock_client):
    mock_client.settings.max_cycles = 2

    plan_with_tasks = Plan(
        tasks=[TaskModel(id="t1", description="task one")],
        rationale="do stuff",
    )
    passed_result = MagicMock(passed=True, summary="ok", failures=[])

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        MockPlanner.return_value.plan = AsyncMock(return_value=plan_with_tasks)
        MockWorker.return_value.run = AsyncMock(return_value="done")
        MockTester.return_value.run = AsyncMock(return_value=passed_result)

        orch = Orchestrator(goal="test", client=mock_client)
        await orch.run()

        assert orch.cycles == 2
