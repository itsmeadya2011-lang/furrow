from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator

# ---------------------------------------------------------------------------
# _is_done() logic tests
# ---------------------------------------------------------------------------


def _orch() -> Orchestrator:
    """Build an Orchestrator without running __init__ (avoids LLM client)."""
    o = Orchestrator.__new__(Orchestrator)
    o.goal = ""
    o.cycles = 0
    o.tasks: list[TaskModel] = []
    o.last_test_passed = None
    o.state_file = None  # type: ignore[assignment]
    return o


def test_is_done_no_tasks():
    o = _orch()
    o.tasks = []
    assert o._is_done() is True


def test_is_done_all_completed():
    o = _orch()
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert o._is_done() is True


def test_is_done_any_failed():
    o = _orch()
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert o._is_done() is False


def test_is_done_pending_tasks():
    o = _orch()
    o.tasks = [TaskModel(id="1", description="a", status="pending")]
    assert o._is_done() is False


# ---------------------------------------------------------------------------
# State persistence tests
# ---------------------------------------------------------------------------


def test_save_and_load_state(tmp_path):
    o = _orch()
    state_file = tmp_path / ".furrow_state.json"
    o.state_file = state_file
    o.goal = "test goal"
    o.cycles = 3
    o.tasks = [
        TaskModel(id="1", description="do thing", status="completed"),
        TaskModel(id="2", description="fix thing", status="failed"),
    ]
    o.last_test_passed = False

    o._save_state()

    o2 = _orch()
    o2.state_file = state_file
    o2._load_state()

    assert o2.goal == "test goal"
    assert o2.cycles == 3
    assert o2.last_test_passed is False
    assert len(o2.tasks) == 2
    assert o2.tasks[0].id == "1"
    assert o2.tasks[0].status == "completed"
    assert o2.tasks[1].id == "2"
    assert o2.tasks[1].status == "failed"


def test_load_state_missing_file(tmp_path):
    o = _orch()
    o.state_file = tmp_path / "nonexistent.json"
    # Should not raise
    o._load_state()
    assert o.tasks == []
    assert o.last_test_passed is None


# ---------------------------------------------------------------------------
# Worker integration (mocked LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_worker_writes_files():
    from furrow.agents.worker import WorkerAgent

    client = MagicMock()
    client.complete = AsyncMock(
        return_value='{"changes": [{"file": "hello.py", "content": "print(123)\\n"}], "summary": "Created hello.py"}'
    )
    client.read_file = AsyncMock(side_effect=FileNotFoundError("no such file"))
    client.write_file = AsyncMock()
    client.list_files = MagicMock(return_value=[])

    task = TaskModel(id="1", description="Create hello.py", files=["hello.py"])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    client.write_file.assert_called_once_with("hello.py", "print(123)\n")
    assert "Created hello.py" in result


@pytest.mark.asyncio
async def test_worker_fallback_on_bad_json():
    from furrow.agents.worker import WorkerAgent

    client = MagicMock()
    client.complete = AsyncMock(return_value="not json at all")
    client.read_file = AsyncMock(side_effect=FileNotFoundError("no such file"))
    client.list_files = MagicMock(return_value=[])

    task = TaskModel(id="1", description="Do something", files=[])
    worker = WorkerAgent(task=task, client=client)
    result = await worker.run()

    assert result == "not json at all"
    client.write_file.assert_not_called()


# ---------------------------------------------------------------------------
# Config model tests (existing + TestResult __test__ flag)
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True
