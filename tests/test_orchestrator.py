"""Tests for the Furrow orchestrator and core components."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import AgentState, CycleRecord

# --------------------------------------------------------------------------- #
# State tests
# --------------------------------------------------------------------------- #


def test_agent_state_defaults():
    state = AgentState(goal="test goal")
    assert state.goal == "test goal"
    assert len(state.cycles) == 0
    assert state.completed_tasks == set()
    assert state.failed_tasks == set()
    assert state.running is False


def test_state_save_and_load(tmp_path: Path):
    state = AgentState(goal="persist me")
    state.cycles.append(
        CycleRecord(
            cycle=1,
            plan={"tasks": []},
            tasks=[TaskModel(id="1", description="do thing")],
            test_result={"passed": True, "summary": "ok", "failures": []},
        )
    )
    state.mark_task_completed("1")
    state.save(tmp_path)

    loaded = AgentState.load(tmp_path)
    assert loaded.goal == "persist me"
    assert len(loaded.cycles) == 1
    assert loaded.cycles[0].cycle == 1
    assert "1" in loaded.completed_tasks
    assert loaded.completed_count == 1


def test_state_load_empty(tmp_path: Path):
    loaded = AgentState.load(tmp_path)
    assert loaded.goal == ""
    assert len(loaded.cycles) == 0


def test_state_add_cycle_updates_status():
    state = AgentState(goal="test")
    tasks = [
        TaskModel(id="1", description="task 1", status="pending"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    state.add_cycle(CycleRecord(cycle=1, tasks=tasks))
    # Initially pending
    assert state.pending_tasks == tasks

    # Mark one completed
    tasks[0].status = "completed"
    state.add_cycle(CycleRecord(cycle=2, tasks=tasks))
    assert "1" in state.completed_tasks
    assert state.completed_count == 1
    assert state.failed_count == 0


def test_state_pending_and_failed_tasks():
    state = AgentState(goal="test")
    state.mark_task_failed("1")
    state.mark_task_completed("2")
    tasks = [
        TaskModel(id="1", description="f", status="failed"),
        TaskModel(id="2", description="c", status="completed"),
        TaskModel(id="3", description="p", status="pending"),
    ]
    state.add_cycle(CycleRecord(cycle=1, tasks=tasks))
    pending = state.pending_tasks
    assert len(pending) == 2  # task 1 (failed) and task 3 (pending)
    assert {t.id for t in pending} == {"1", "3"}


# --------------------------------------------------------------------------- #
# Orchestrator tests
# --------------------------------------------------------------------------- #


def test_orchestrator_is_done_with_no_tasks():
    """Orchestrator should report done when planner returns no tasks."""
    mock_client = MagicMock()
    mock_client.settings = MagicMock()
    mock_client.settings.max_parallel_tasks = 5
    mock_client.settings.max_cycles = 0
    mock_client.settings.workspace = Path("/tmp")

    orch = Orchestrator(goal="nothing to do", client=mock_client)
    orch.current_plan = Plan(tasks=[], rationale="nothing")
    orch.state.cycles.append(CycleRecord(cycle=1, tasks=[]))
    assert orch._is_done() is True


def test_orchestrator_not_done_with_pending_tasks():
    mock_client = MagicMock()
    mock_client.settings = MagicMock()
    mock_client.settings.max_parallel_tasks = 5
    mock_client.settings.max_cycles = 0
    mock_client.settings.workspace = Path("/tmp")

    orch = Orchestrator(goal="do something", client=mock_client)
    orch.current_plan = Plan(
        tasks=[TaskModel(id="1", description="task")],
        rationale="plan",
    )
    orch.state.cycles.append(
        CycleRecord(
            cycle=1,
            tasks=[TaskModel(id="1", description="task", status="pending")],
        )
    )
    assert orch._is_done() is False


def test_orchestrator_get_tasks_returns_tracked():
    mock_client = MagicMock()
    mock_client.settings = MagicMock()
    mock_client.settings.max_parallel_tasks = 5
    mock_client.settings.max_cycles = 0
    mock_client.settings.workspace = Path("/tmp")

    orch = Orchestrator(goal="test", client=mock_client)
    tasks = [TaskModel(id="1", description="t1"), TaskModel(id="2", description="t2")]
    orch.state.add_cycle(CycleRecord(cycle=1, tasks=tasks))
    retrieved = orch._get_tasks()
    assert len(retrieved) == 2
    assert retrieved[0].id == "1"
    assert retrieved[1].id == "2"


@pytest.mark.asyncio
async def test_orchestrator_max_cycles_limit():
    """Orchestrator should stop after max_cycles."""
    mock_client = MagicMock()
    mock_client.settings = MagicMock()
    mock_client.settings.max_parallel_tasks = 5
    mock_client.settings.max_cycles = 1
    mock_client.settings.workspace = Path("/tmp")
    mock_client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "done"}')

    # Mock planner to return no tasks (so it won't loop)
    with patch.object(Orchestrator, "_cycle", new_callable=AsyncMock):
        orch = Orchestrator(goal="limited", client=mock_client)
        orch._cycle = AsyncMock()
        orch._is_done = MagicMock(return_value=False)
        await orch.run()
        assert orch.cycles == 1
