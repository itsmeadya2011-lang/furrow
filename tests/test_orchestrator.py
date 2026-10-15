from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_orchestrator(plan_tasks=None, test_result=None, max_cycles=0, max_parallel=5):
    client = MagicMock()
    client.settings.max_cycles = max_cycles
    client.settings.max_parallel_tasks = max_parallel

    planner = MagicMock()
    planner.plan = AsyncMock(return_value=Plan(tasks=plan_tasks or [], rationale="test"))

    worker_run = AsyncMock(return_value="done")

    tester_run = AsyncMock(return_value=test_result or TestResult(passed=True, summary="ok"))

    orch = Orchestrator(goal="test goal", client=client)
    orch.planner = planner

    import furrow.core.orchestrator as orch_mod
    orig_worker = orch_mod.WorkerAgent
    orig_tester = orch_mod.TesterAgent
    orch_mod.WorkerAgent = MagicMock(return_value=MagicMock(run=worker_run))
    orch_mod.TesterAgent = MagicMock(return_value=MagicMock(run=tester_run))

    return orch, planner, worker_run, tester_run, (orig_worker, orig_tester)


def test_orchestrator_done_after_one_cycle():
    orch, *_ = _make_orchestrator(
        plan_tasks=[TaskModel(id="1", description="task one")],
        test_result=TestResult(passed=True, summary="all good"),
    )
    import asyncio
    asyncio.run(orch.run())
    assert orch.cycles == 1


def test_orchestrator_loops_on_failure():
    """When tests fail, the loop continues and the goal is rewritten."""
    orch, *_ = _make_orchestrator(
        plan_tasks=[TaskModel(id="1", description="task one")],
        test_result=TestResult(passed=False, summary="broken", failures=["fix this"]),
        max_cycles=3,
    )
    import asyncio
    asyncio.run(orch.run())
    assert orch.cycles == 3
    assert "Fix failing tests" in orch.goal


def test_orchestrator_max_cycles_halts():
    """max_cycles=0 means unlimited; max_cycles=N halts after N cycles."""
    orch, *_ = _make_orchestrator(
        plan_tasks=[TaskModel(id="1", description="task one")],
        test_result=TestResult(passed=False, summary="broken", failures=["fix this"]),
        max_cycles=2,
    )
    import asyncio
    asyncio.run(orch.run())
    assert orch.cycles == 2


def test_orchestrator_no_tasks_is_done():
    """If the planner returns no tasks, the orchestrator halts after one cycle."""
    orch, *_ = _make_orchestrator(plan_tasks=[], test_result=TestResult(passed=True, summary="ok"))
    import asyncio
    asyncio.run(orch.run())
    assert orch.cycles == 1


def test_orchestrator_get_tasks_returns_stored():
    """_get_tasks returns the tasks from the current cycle."""
    orch, *_ = _make_orchestrator(plan_tasks=[
        TaskModel(id="1", description="a"),
        TaskModel(id="2", description="b"),
    ])
    import asyncio
    asyncio.run(orch._cycle())
    assert len(orch._get_tasks()) == 2


def test_orchestrator_failed_task_still_halts_when_tests_pass():
    """If a task fails but tests pass, the orchestrator halts (task failure
    is a task-level issue, not something the loop fixes by re-planning)."""
    orch, planner, worker_run, *_ = _make_orchestrator(
        plan_tasks=[TaskModel(id="1", description="task one")],
        test_result=TestResult(passed=True, summary="ok"),
    )
    worker_run.side_effect = Exception("boom")
    import asyncio
    asyncio.run(orch.run())
    assert orch.cycles == 1