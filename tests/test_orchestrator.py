"""Tests for the Orchestrator: cycle loop, max_cycles, task tracking, and concurrency."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def _make_settings(**overrides):
    """Create Settings with test defaults."""
    kwargs = dict(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        planner_model="claude-3-5-haiku-20241022",
        worker_model="claude-3-5-sonnet-20241022",
        tester_model="claude-3-5-sonnet-20241022",
        anthropic_api_key="sk-test-anthropic",
        max_parallel_tasks=2,
        max_cycles=0,
        workspace=Path.cwd(),
        log_level="INFO",
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


def _make_plan_response(tasks):
    """Build a JSON plan response string from a list of task dicts."""
    import json
    return json.dumps({"tasks": tasks, "rationale": "test plan"})


@pytest.mark.asyncio
async def test_orchestrator_terminates_on_no_tasks():
    """If the planner returns no tasks, the orchestrator should stop."""
    settings = _make_settings()
    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value='{"tasks": [], "rationale": "nothing to do"}')

    orch = Orchestrator(goal="do nothing", client=client, settings=settings)
    await orch.run()

    assert orch.cycles == 1
    assert orch.all_tasks == []


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles():
    """Orchestrator should halt after max_cycles even if tests fail."""
    settings = _make_settings(max_cycles=1)

    plan_tasks = [
        {"id": "1", "description": "task 1", "files": [], "dependencies": []},
    ]

    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(
        return_value=_make_plan_response(plan_tasks)
    )

    orch = Orchestrator(goal="test goal", client=client, settings=settings)
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_tester = MockTester.return_value
        mock_tester.run = AsyncMock(
            return_value=MagicMock(passed=False, summary="fail", failures=["something broke"])
        )
        await orch.run()

    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_orchestrator_tracks_completed_tasks():
    """Tasks should be tracked in all_tasks and marked as completed."""
    settings = _make_settings()
    plan_tasks = [
        {"id": "1", "description": "task 1", "files": [], "dependencies": []},
        {"id": "2", "description": "task 2", "files": [], "dependencies": []},
    ]

    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value=_make_plan_response(plan_tasks))

    orch = Orchestrator(goal="test goal", client=client, settings=settings)
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_tester = MockTester.return_value
        mock_tester.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="all good", failures=[])
        )
        await orch.run()

    assert len(orch.all_tasks) == 2
    assert all(t.status == "completed" for t in orch.all_tasks)
    assert orch._is_done()  # Should be done since no failed tasks and all completed


@pytest.mark.asyncio
async def test_orchestrator_is_done_with_failures():
    """_is_done should return False if any task failed."""
    settings = _make_settings()

    plan_tasks = [
        {"id": "1", "description": "task 1", "files": [], "dependencies": []},
    ]

    client = MagicMock()
    client.settings = settings
    # First call returns the plan, worker calls return different things
    client.complete = AsyncMock()
    # Plan call
    client.complete.side_effect = [
        _make_plan_response(plan_tasks),  # planner
        "worker result 1",  # worker
        '{"passed": true, "summary": "ok", "failures": []}',  # tester
    ]

    orch = Orchestrator(goal="test goal", client=client, settings=settings)
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_tester = MockTester.return_value
        mock_tester.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="ok", failures=[])
        )
        await orch.run()

    # Simulate a failed task
    orch.all_tasks[0].status = "failed"
    assert orch._is_done() is False

    # Reset and check all completed
    orch.all_tasks[0].status = "completed"
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_orchestrator_respects_dependencies():
    """Tasks with dependencies should not run before their dependencies."""
    settings = _make_settings(max_parallel_tasks=1)
    plan_tasks = [
        {"id": "1", "description": "task 1", "files": [], "dependencies": []},
        {"id": "2", "description": "task 2", "files": [], "dependencies": ["1"]},
    ]

    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value=_make_plan_response(plan_tasks))

    orch = Orchestrator(goal="test goal", client=client, settings=settings)
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_tester = MockTester.return_value
        mock_tester.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="ok", failures=[])
        )
        await orch.run()

    assert len(orch.all_tasks) == 2
    assert all(t.status == "completed" for t in orch.all_tasks)


@pytest.mark.asyncio
async def test_orchestrator_handles_worker_exception():
    """A failing worker should mark the task as failed, not crash."""
    settings = _make_settings()
    plan_tasks = [
        {"id": "1", "description": "task 1", "files": [], "dependencies": []},
    ]

    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value=_make_plan_response(plan_tasks))

    orch = Orchestrator(goal="test goal", client=client, settings=settings)
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_tester = MockTester.return_value
        mock_tester.run = AsyncMock(
            return_value=MagicMock(passed=True, summary="ok", failures=[])
        )
        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
            mock_worker = MockWorker.return_value
            mock_worker.run = AsyncMock(side_effect=RuntimeError("worker boom"))
            await orch.run()

    assert orch.all_tasks[0].status == "failed"
    assert "worker boom" in orch.all_tasks[0].result


@pytest.mark.asyncio
async def test_orchestrator_cycles_on_test_failure():
    """If tests fail, the orchestrator should loop and try again (up to max_cycles)."""
    settings = _make_settings(max_cycles=3)
    plan_tasks = [
        {"id": "1", "description": "task 1", "files": [], "dependencies": []},
    ]

    client = MagicMock()
    client.settings = settings
    client.complete = AsyncMock(return_value=_make_plan_response(plan_tasks))

    orch = Orchestrator(goal="test goal", client=client, settings=settings)
    with patch("furrow.core.orchestrator.TesterAgent") as MockTester:
        mock_tester = MockTester.return_value
        # First cycle: tests fail. Second cycle: tests pass.
        mock_tester.run = AsyncMock(
            side_effect=[
                MagicMock(passed=False, summary="fail", failures=["bug 1"]),
                MagicMock(passed=True, summary="ok", failures=[]),
            ]
        )
        await orch.run()

    assert orch.cycles == 2
    assert orch._is_done()
