from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def test_get_tasks_returns_stored_tasks():
    orchestrator = Orchestrator(goal="test")
    task = TaskModel(id="1", description="task")
    orchestrator.tasks = [task]
    assert orchestrator._get_tasks() == [task]


def test_is_done_all_completed():
    orchestrator = Orchestrator(goal="test")
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orchestrator._is_done() is True


def test_is_done_has_failed():
    orchestrator = Orchestrator(goal="test")
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_mixed_statuses():
    orchestrator = Orchestrator(goal="test")
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_empty_tasks():
    orchestrator = Orchestrator(goal="test")
    assert orchestrator._is_done() is True


async def test_max_cycles_enforced(tmp_path):
    state_file = tmp_path / "state.json"
    settings = Settings(max_cycles=1, state_file=state_file)
    mock_client = MagicMock()
    orchestrator = Orchestrator(goal="test", client=mock_client, settings=settings)
    orchestrator.tasks = [TaskModel(id="1", description="pending")]

    with patch.object(orchestrator, '_cycle', new_callable=AsyncMock) as mock_cycle:
        await orchestrator.run()
        mock_cycle.assert_called_once()
        assert orchestrator.cycles == 1


async def test_state_save_load(tmp_path):
    state_file = tmp_path / "state.json"
    settings = Settings(state_file=state_file)
    orchestrator = Orchestrator(goal="original goal", settings=settings)
    orchestrator.tasks = [TaskModel(id="1", description="task1", status="completed")]
    orchestrator.cycles = 3
    await orchestrator._save_state()

    orchestrator2 = Orchestrator(goal="original goal", settings=settings)
    await orchestrator2._load_state()
    assert orchestrator2.goal == "original goal"
    assert orchestrator2.cycles == 3
    assert len(orchestrator2.tasks) == 1
    assert orchestrator2.tasks[0].id == "1"
    assert orchestrator2.tasks[0].status == "completed"


def test_console_injected():
    mock_console = MagicMock()
    orchestrator = Orchestrator(goal="test", console=mock_console)
    assert orchestrator.console is mock_console
