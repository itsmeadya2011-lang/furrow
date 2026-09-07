import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_is_done_all_completed():
    orchestrator = Orchestrator(goal="test", client=MagicMock())
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orchestrator._is_done() is True


def test_is_done_with_failed():
    orchestrator = Orchestrator(goal="test", client=MagicMock())
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_partial():
    orchestrator = Orchestrator(goal="test", client=MagicMock())
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_get_tasks_empty():
    orchestrator = Orchestrator(goal="test", client=MagicMock())
    assert orchestrator._get_tasks() == []


@pytest.mark.asyncio
async def test_max_cycles_enforced(tmp_path):
    settings = Settings(max_cycles=1)
    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner, \
         patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
         patch("furrow.core.orchestrator.TesterAgent") as MockTester:

        mock_planner = MockPlanner.return_value
        mock_planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="ok"))

        mock_worker = MockWorker.return_value
        mock_worker.run = AsyncMock(return_value="done")

        mock_tester = MockTester.return_value
        mock_tester.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

        orchestrator = Orchestrator(
            goal="test", client=MagicMock(), state_path=tmp_path / "state.json"
        )
        orchestrator.settings = settings

        await orchestrator.run()

        assert orchestrator.cycles == 1
        mock_planner.plan.assert_called_once()


def test_emits_output():
    callback = MagicMock()
    orchestrator = Orchestrator(goal="test", client=MagicMock(), on_output=callback)
    orchestrator._emit("hello")
    callback.assert_called_once_with("hello")


def test_save_state_creates_file(tmp_path):
    orchestrator = Orchestrator(goal="test", client=MagicMock(), state_path=tmp_path / "state.json")
    orchestrator.current_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    orchestrator.cycles = 2
    orchestrator._save_state()
    assert (tmp_path / "state.json").exists()


@pytest.mark.asyncio
async def test_state_loaded_on_run(tmp_path):
    from furrow.config import FurrowState
    settings = Settings(max_cycles=5)
    state = FurrowState(
        goal="loaded goal",
        cycle=1,
        tasks=[TaskModel(id="1", description="a", status="completed")],
        completed=["1"],
        failed=[],
    )
    state.save(tmp_path / "state.json")

    with patch("furrow.core.orchestrator.PlannerAgent") as MockPlanner:
        mock_planner = MockPlanner.return_value
        mock_planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="ok"))

        orchestrator = Orchestrator(
            goal="original goal", client=MagicMock(), state_path=tmp_path / "state.json"
        )
        orchestrator.settings = settings
        await orchestrator.run()
        assert orchestrator.goal == "loaded goal"
        assert orchestrator.cycles == 2  # started at 1 from loaded state, incremented to 2
        mock_planner.plan.assert_called_once()
