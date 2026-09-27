import json
from unittest.mock import AsyncMock, MagicMock

from furrow.config import Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_is_done_no_tasks_returns_true():
    orch = Orchestrator(goal="g", client=MagicMock())
    orch.tasks = []
    assert orch._is_done() is True


def test_is_done_tasks_not_all_completed_returns_false():
    orch = Orchestrator(goal="g", client=MagicMock())
    orch.tasks = [TaskModel(id="1", description="a", status="pending")]
    assert orch._is_done() is False


def test_is_done_all_completed_tests_passed_returns_true():
    orch = Orchestrator(goal="g", client=MagicMock())
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


def test_is_done_all_completed_tests_failed_returns_false():
    orch = Orchestrator(goal="g", client=MagicMock())
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    orch._last_test_result = TestResult(passed=False, summary="bad")
    assert orch._is_done() is False


def test_get_tasks_returns_stored_tasks():
    orch = Orchestrator(goal="g", client=MagicMock())
    tasks = [TaskModel(id="1", description="a")]
    orch.tasks = tasks
    assert orch._get_tasks() == tasks


def test_save_state_writes_json(tmp_path):
    client = MagicMock()
    client.settings = Settings(workspace=tmp_path)
    orch = Orchestrator(goal="g", client=client)
    orch.tasks = [TaskModel(id="1", description="a", status="completed")]
    orch._last_test_result = TestResult(passed=True, summary="ok")
    orch._save_state()
    state_file = tmp_path / ".furrow_state.json"
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert data["goal"] == "g"
    assert data["tasks"][0]["id"] == "1"
    assert data["test_result"]["passed"] is True


async def test_max_cycles_halts_run(tmp_path):
    client = MagicMock()
    client.settings = Settings(workspace=tmp_path, max_cycles=1)
    orch = Orchestrator(goal="g", client=client)
    orch._cycle = AsyncMock()
    orch._is_done = lambda: False
    await orch.run()
    assert orch.cycles == 2
    assert orch._cycle.await_count == 1
