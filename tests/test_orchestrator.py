"""Tests for the Orchestrator's task lifecycle and cycle management."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel
from furrow.core.orchestrator import Orchestrator


def _make_mock_client(
    max_parallel_tasks: int = 5,
    max_cycles: int = 0,
    max_tokens: int = 4096,
):
    """Create a mock LLMClient with configurable settings."""
    client = MagicMock()
    client.settings = MagicMock(
        max_parallel_tasks=max_parallel_tasks,
        max_cycles=max_cycles,
        max_tokens=max_tokens,
    )
    return client


# ---------------------------------------------------------------------------
# _get_tasks / _merge_tasks
# ---------------------------------------------------------------------------

def test_get_tasks_returns_accumulated_tasks():
    """_get_tasks must return the internal task list, not an empty list."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    assert orch._get_tasks() == []
    orch.tasks.append(TaskModel(id="1", description="a"))
    assert len(orch._get_tasks()) == 1


def test_merge_tasks_adds_new_tasks():
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    new = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    orch._merge_tasks(new)
    assert len(orch.tasks) == 2
    assert orch.tasks[0].id == "1"
    assert orch.tasks[1].id == "2"


def test_merge_tasks_keeps_completed_tasks():
    """Completed tasks should not be modified or re-added."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    existing = TaskModel(id="1", description="old", status="completed")
    orch.tasks.append(existing)
    new = [TaskModel(id="1", description="updated")]
    orch._merge_tasks(new)
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"
    assert orch.tasks[0].description == "old"


def test_merge_tasks_resets_failed_tasks():
    """Failed tasks should be reset to pending when re-planned."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    existing = TaskModel(id="1", description="old", status="failed")
    orch.tasks.append(existing)
    new = [TaskModel(id="1", description="updated")]
    orch._merge_tasks(new)
    assert orch.tasks[0].status == "pending"
    assert orch.tasks[0].description == "updated"


def test_merge_tasks_updates_pending_tasks():
    """Pending tasks should be updated with latest plan data."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    existing = TaskModel(
        id="1", description="old", status="pending", files=["a.py"]
    )
    orch.tasks.append(existing)
    new = [
        TaskModel(
            id="1",
            description="updated",
            files=["b.py"],
            dependencies=["2"],
        )
    ]
    orch._merge_tasks(new)
    assert len(orch.tasks) == 1
    assert orch.tasks[0].description == "updated"
    assert orch.tasks[0].files == ["b.py"]
    assert orch.tasks[0].dependencies == ["2"]


# ---------------------------------------------------------------------------
# _get_runnable_tasks / dependencies
# ---------------------------------------------------------------------------

def test_get_runnable_tasks_respects_dependencies():
    """Tasks with unmet dependencies must not be runnable."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="dep1", status="pending"),
        TaskModel(id="2", description="dep2", status="pending"),
        TaskModel(
            id="3", description="child",
            dependencies=["1", "2"], status="pending",
        ),
    ]
    # Only tasks 1 and 2 are runnable; task 3 has unmet deps.
    runnable = orch._get_runnable_tasks()
    runnable_ids = {t.id for t in runnable}
    assert runnable_ids == {"1", "2"}

    # After completing tasks 1 and 2, task 3 becomes runnable.
    orch.tasks[0].status = "completed"
    orch.tasks[1].status = "completed"
    runnable = orch._get_runnable_tasks()
    assert len(runnable) == 1
    assert runnable[0].id == "3"


def test_get_runnable_tasks_excludes_completed_and_failed():
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="done", status="completed"),
        TaskModel(id="2", description="fail", status="failed"),
        TaskModel(id="3", description="ok", status="pending"),
    ]
    runnable = orch._get_runnable_tasks()
    assert len(runnable) == 1
    assert runnable[0].id == "3"


# ---------------------------------------------------------------------------
# _is_done
# ---------------------------------------------------------------------------

def test_is_done_empty_tasks():
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    assert orch._is_done() is True


def test_is_done_all_completed():
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_mixed_terminal_and_pending():
    """Failed tasks are terminal; the loop continues only if pending tasks exist."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
        TaskModel(id="3", description="c", status="pending"),
    ]
    assert orch._is_done() is False


def test_is_done_all_terminal_states():
    """All terminal states (completed/failed/skipped) → done."""
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
        TaskModel(id="3", description="c", status="skipped"),
    ]
    assert orch._is_done() is True


# ---------------------------------------------------------------------------
# max_cycles enforcement
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_cycles_respected():
    """When max_cycles is set, the loop must halt after that many cycles."""
    client = _make_mock_client(max_cycles=2)
    orch = Orchestrator(goal="test", client=client)

    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(
        return_value=MagicMock(passed=True, summary="", failures=[])
    )
    # Force _is_done to return False so the loop only exits via max_cycles.
    orch._is_done = MagicMock(return_value=False)

    await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_max_cycles_zero_unlimited():
    """max_cycles=0 means unlimited; a plan that returns no tasks should halt."""
    client = _make_mock_client(max_cycles=0)
    orch = Orchestrator(goal="test", client=client)

    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(
        return_value=MagicMock(passed=True, summary="", failures=[])
    )

    await orch.run()
    assert orch.cycles == 1  # halts because no tasks


# ---------------------------------------------------------------------------
# semaphore / max_parallel_tasks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_parallel_tasks_enforced():
    """At most max_parallel_tasks should run concurrently."""
    settings = MagicMock(
        max_parallel_tasks=2, max_cycles=0, max_tokens=4096,
        provider="anthropic"
    )
    client = MagicMock()
    client.settings = settings
    orch = Orchestrator(goal="test", client=client)

    current = 0
    peak = 0

    async def fake_run():
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return "done"

    orch.tasks = [
        TaskModel(id=str(i), description=f"t{i}", status="pending")
        for i in range(1, 6)
    ]

    runnable = orch._get_runnable_tasks()
    assert len(runnable) == 5

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(side_effect=fake_run)
        MockWorker.return_value = mock_instance
        results = await asyncio.gather(
            *(orch._run_task(t) for t in runnable)
        )

    assert len(results) == 5
    assert peak <= 2


# ---------------------------------------------------------------------------
# _cycle integration
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cycle_merges_and_executes_tasks():
    """A single cycle should plan, merge, execute, and test."""
    client = _make_mock_client()
    orch = Orchestrator(goal="build feature", client=client)

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="implement auth"),
            TaskModel(id="2", description="write tests"),
        ],
        rationale="auth and tests",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(
        return_value=MagicMock(passed=True, summary="ok", failures=[])
    )

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value="completed")
        MockWorker.return_value = mock_instance
        await orch._cycle()

    task_ids = {t.id for t in orch.tasks}
    assert task_ids == {"1", "2"}
    assert all(t.status == "completed" for t in orch.tasks)


@pytest.mark.asyncio
async def test_cycle_no_runnable_when_deps_unmet():
    """Tasks with unmet dependencies should not be executed."""
    client = _make_mock_client()
    orch = Orchestrator(goal="build feature", client=client)

    plan = Plan(
        tasks=[
            TaskModel(
                id="1", description="implement auth",
                dependencies=["2"],
            ),
            TaskModel(id="2", description="write tests"),
        ],
        rationale="dep graph",
    )
    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=plan)

    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(
        return_value=MagicMock(passed=True, summary="ok", failures=[])
    )

    with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker:
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value="completed")
        MockWorker.return_value = mock_instance
        await orch._cycle()

    # Task 2 has no deps → runnable.  Task 1 depends on 2 → not runnable.
    ids_started = {t.id for t in orch.tasks if t.status == "completed"}
    assert ids_started == {"2"}
    assert orch.tasks[0].status == "pending"  # task 1 still pending


@pytest.mark.asyncio
async def test_tests_pass_flag_set_on_success():
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)

    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(
        return_value=MagicMock(passed=True, summary="ok", failures=[])
    )

    await orch._cycle()
    assert orch._tests_passed is True


@pytest.mark.asyncio
async def test_tests_pass_flag_cleared_on_failure():
    client = _make_mock_client()
    orch = Orchestrator(goal="test", client=client)

    orch.planner = MagicMock()
    orch.planner.plan = AsyncMock(return_value=Plan(tasks=[], rationale="done"))
    orch.tester = MagicMock()
    orch.tester.run = AsyncMock(
        return_value=MagicMock(
            passed=False, summary="boom", failures=["assertion failed"]
        )
    )

    await orch._cycle()
    assert orch._tests_passed is False
    # Goal should be rewritten to fix failures.
    assert "Fix failing tests" in orch.goal
