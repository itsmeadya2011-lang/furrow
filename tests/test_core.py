import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult, Settings, Provider
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import FurrowState, StateManager
from furrow.llm import LLMClient


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_done_when_all_completed():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.last_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orchestrator.cycles = 1
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._is_done() is True


def test_orchestrator_not_done_when_failed():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.last_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    orchestrator.cycles = 1
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._is_done() is False


def test_orchestrator_not_done_when_pending():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.last_tasks = [
        TaskModel(id="1", description="a", status="pending"),
    ]
    orchestrator.cycles = 1
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._is_done() is False


def test_orchestrator_done_when_no_tasks():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.last_tasks = []
    orchestrator.cycles = 1
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    assert orchestrator._is_done() is True


def test_orchestrator_stops_at_max_cycles():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.last_tasks = [
        TaskModel(id="1", description="a", status="pending"),
    ]
    orchestrator.cycles = 5
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 5
    assert orchestrator._is_done() is True


def test_orchestrator_continues_below_max_cycles():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.last_tasks = [
        TaskModel(id="1", description="a", status="pending"),
    ]
    orchestrator.cycles = 3
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 10
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_cycle_sets_last_tasks():
    orchestrator = Orchestrator.__new__(Orchestrator)
    orchestrator.goal = "test goal"
    orchestrator.client = MagicMock()
    orchestrator.client.settings.max_cycles = 0
    orchestrator.planner = AsyncMock()
    test_plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
    orchestrator.planner.plan.return_value = test_plan

    mock_worker_instance = AsyncMock()
    mock_worker_instance.run.return_value = "done"
    with patch("furrow.core.orchestrator.WorkerAgent", return_value=mock_worker_instance):
        mock_tester_instance = AsyncMock()
        mock_tester_instance.run.return_value = TestResult(passed=True, summary="ok", failures=[])
        with patch("furrow.core.orchestrator.TesterAgent", return_value=mock_tester_instance):
            await orchestrator._cycle()

    assert len(orchestrator.last_tasks) == 1
    assert orchestrator.last_tasks[0].id == "1"
    assert orchestrator.last_tasks[0].status == "completed"
    assert orchestrator.last_tasks[0].result == "done"


@pytest.mark.asyncio
async def test_llm_client_complete_retry():
    settings = Settings(provider=Provider.ANTHROPIC, model="test-model")
    client = LLMClient(settings=settings)
    with patch.object(client, "_complete_anthropic", new_callable=AsyncMock) as mock_complete:
        mock_complete.return_value = "success"
        result = await client.complete("hello", model="test-model")
        assert result == "success"
        mock_complete.assert_called_once()


def test_state_manager_save_and_load(tmp_path):
    manager = StateManager(workspace=tmp_path)
    state = FurrowState(goal="test", cycles=2, last_tasks=[{"id": "1", "description": "a", "status": "completed"}])
    manager.save(state)
    loaded = manager.load()
    assert loaded is not None
    assert loaded.goal == "test"
    assert loaded.cycles == 2
    assert loaded.last_tasks[0]["id"] == "1"


def test_state_manager_load_missing(tmp_path):
    manager = StateManager(workspace=tmp_path)
    assert manager.load() is None
