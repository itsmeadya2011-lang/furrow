"""Tests for the Furrow orchestrator's task tracking and cycle management."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def _make_task(status: str = "pending") -> TaskModel:
    return TaskModel(id="1", description="do thing", status=status)


def _make_plan(tasks=None) -> Plan:
    if tasks is None:
        tasks = [_make_task()]
    return Plan(tasks=tasks, rationale="test plan")


@pytest.fixture
def mock_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def mock_planner(mock_client: MagicMock) -> MagicMock:
    planner = MagicMock()
    planner.plan = AsyncMock(return_value=_make_plan())
    return planner


@pytest.fixture
def mock_tester(mock_client: MagicMock) -> MagicMock:
    tester = MagicMock()
    tester.run = AsyncMock(return_value=MagicMock(passed=True, summary="ok", failures=[]))
    return tester


@pytest.mark.asyncio
async def test_is_done_all_completed(mock_planner, mock_tester):
    """Orchestrator should be done when all tasks are completed."""
    orch = Orchestrator(goal="test goal")
    orch.planner = mock_planner
    orch.tester = mock_tester
    plan = _make_plan([TaskModel(id="1", description="task 1", status="completed")])
    orch.current_plan = plan
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_with_failures(mock_planner, mock_tester):
    """Orchestrator should NOT be done when there are failed tasks (they need retry)."""
    orch = Orchestrator(goal="test goal")
    orch.planner = mock_planner
    orch.tester = mock_tester
    plan = _make_plan([TaskModel(id="1", description="task 1", status="failed")])
    orch.current_plan = plan
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_no_plan(mock_planner, mock_tester):
    """Orchestrator should be done when there is no current plan."""
    orch = Orchestrator(goal="test goal")
    orch.planner = mock_planner
    orch.tester = mock_tester
    assert orch.current_plan is None
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_empty_plan(mock_planner, mock_tester):
    """Orchestrator should be done when the plan has no tasks."""
    orch = Orchestrator(goal="test goal")
    orch.planner = mock_planner
    orch.tester = mock_tester
    plan = _make_plan([])
    orch.current_plan = plan
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_get_tasks_returns_current_plan_tasks(mock_planner, mock_tester):
    """_get_tasks should return tasks from the current plan."""
    orch = Orchestrator(goal="test goal")
    orch.planner = mock_planner
    orch.tester = mock_tester
    plan = _make_plan([
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ])
    orch.current_plan = plan
    tasks = orch._get_tasks()
    assert len(tasks) == 2


@pytest.mark.asyncio
async def test_get_tasks_empty_when_no_plan(mock_planner, mock_tester):
    """_get_tasks should return empty list when no plan exists."""
    orch = Orchestrator(goal="test goal")
    orch.planner = mock_planner
    orch.tester = mock_tester
    assert orch._get_tasks() == []


@pytest.mark.asyncio
async def test_max_cycles_enforced(mock_planner, mock_tester):
    """Orchestrator should stop after reaching max_cycles, even if not done."""
    custom_settings = Settings(max_cycles=2)
    orch = Orchestrator(goal="test goal", settings=custom_settings)
    orch.planner = mock_planner
    orch.tester = mock_tester

    call_count = 0

    async def fake_cycle(goal: str) -> None:
        nonlocal call_count
        call_count += 1
        # Set failed tasks so _is_done returns False, forcing max_cycles to stop
        orch.current_plan = _make_plan([TaskModel(id="1", description="task", status="failed")])

    orch._cycle = fake_cycle

    await orch.run()

    assert orch.cycles == 2
    assert call_count == 2


@pytest.mark.asyncio
async def test_max_cycles_zero_unlimited(mock_planner, mock_tester):
    """max_cycles=0 means unlimited; orchestrator runs until goal is done."""
    custom_settings = Settings(max_cycles=0)
    orch = Orchestrator(goal="test goal", settings=custom_settings)
    orch.planner = mock_planner
    orch.tester = mock_tester

    async def fake_cycle(goal: str) -> None:
        # Simulate all tasks completed on first cycle
        orch.current_plan = _make_plan([TaskModel(id="1", description="task", status="completed")])

    orch._cycle = fake_cycle

    await orch.run()

    assert orch.cycles == 1  # Stopped because goal completed, not because of max_cycles


@pytest.mark.asyncio
async def test_output_sink_receives_messages(mock_planner, mock_tester):
    """Orchestrator should send messages to the output sink when set."""
    received = []

    def sink(msg: str) -> None:
        received.append(msg)

    orch = Orchestrator(goal="test goal", output_sink=sink)
    orch.planner = mock_planner
    orch.tester = mock_tester
    orch._cycle = AsyncMock()

    await orch.run()

    assert len(received) > 0
    assert any("Furrow" in msg for msg in received)


@pytest.mark.asyncio
async def test_add_input_stored(mock_planner, mock_tester):
    """add_input should store mid-loop input for the next cycle."""
    orch = Orchestrator(goal="test goal")
    orch.planner = mock_planner
    orch.tester = mock_tester
    orch._cycle = AsyncMock()

    orch.add_input("new goal")
    assert orch._mid_loop_input == "new goal"


@pytest.mark.asyncio
async def test_run_stops_on_done(mock_planner, mock_tester):
    """Orchestrator should stop when _is_done returns True after cycle."""
    orch = Orchestrator(goal="test goal")
    orch.planner = mock_planner
    orch.tester = mock_tester

    async def fake_cycle(goal: str) -> None:
        orch.current_plan = _make_plan([TaskModel(id="1", description="task", status="completed")])

    orch._cycle = fake_cycle

    await orch.run()
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_run_continues_when_not_done(mock_planner, mock_tester):
    """Orchestrator should continue cycling when tasks are not all complete."""
    orch = Orchestrator(goal="test goal", settings=Settings(max_cycles=2))
    orch.planner = mock_planner
    orch.tester = mock_tester

    async def fake_cycle(goal: str) -> None:
        orch.current_plan = _make_plan([TaskModel(id="1", description="task", status="failed")])

    orch._cycle = fake_cycle

    await orch.run()
    assert orch.cycles == 2
