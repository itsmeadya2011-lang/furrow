from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def make_settings(**kwargs: Any) -> Settings:
    return Settings(**kwargs)


def test_orchestrator_is_done_with_no_tasks() -> None:
    o = Orchestrator(goal="test")
    assert o._is_done() is False


def test_orchestrator_is_done_all_completed() -> None:
    o = Orchestrator(goal="test")
    o.tasks = [TaskModel(id="1", description="do thing", status="completed")]
    assert o._is_done() is True


def test_orchestrator_is_done_with_failed_task() -> None:
    o = Orchestrator(goal="test")
    o.tasks = [TaskModel(id="1", description="do thing", status="failed")]
    assert o._is_done() is False


def test_orchestrator_is_done_partial_completion() -> None:
    o = Orchestrator(goal="test")
    o.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert o._is_done() is False


def test_orchestrator_max_cycles_enforced() -> None:
    settings = make_settings(max_cycles=2)
    client = MagicMock()
    client.settings = settings
    o = Orchestrator(goal="test", client=client)
    o._cycle = AsyncMock()
    o._is_done = MagicMock(return_value=False)
    asyncio.run(o.run())
    assert o.cycles == 2


def test_orchestrator_stops_when_done() -> None:
    o = Orchestrator(goal="test")
    o._cycle = AsyncMock()
    o._is_done = MagicMock(side_effect=[False, True])
    asyncio.run(o.run())
    assert o.cycles == 2


def test_orchestrator_emit_does_not_raise_on_no_callback() -> None:
    o = Orchestrator(goal="test")
    o._emit("hello")  # should not raise


def test_orchestrator_emit_with_sync_callback() -> None:
    messages: list[str] = []
    o = Orchestrator(goal="test", on_output=messages.append)
    o._emit("hello")
    assert messages == ["hello"]


def test_orchestrator_emit_with_async_callback() -> None:
    messages: list[str] = []

    async def cb(msg: str) -> None:
        messages.append(msg)

    o = Orchestrator(goal="test", on_output=cb)
    o._emit("hello")
    assert messages == ["hello"]


def test_orchestrator_get_tasks_returns_stored() -> None:
    o = Orchestrator(goal="test")
    tasks = [TaskModel(id="1", description="x")]
    o.tasks = tasks
    assert o._get_tasks() is tasks


def test_orchestrator_cycle_stores_tasks() -> None:
    o = Orchestrator(goal="test")
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="because",
    )
    o.planner = MagicMock()
    o.planner.plan = AsyncMock(return_value=plan)
    o.client = MagicMock()
    o.client.settings = make_settings()

    import furrow.core.orchestrator as orch_mod
    original = orch_mod.WorkerAgent
    orch_mod.WorkerAgent = MagicMock()
    orch_mod.WorkerAgent.return_value.run = AsyncMock(return_value="done")

    try:
        asyncio.run(o._cycle())
    finally:
        orch_mod.WorkerAgent = original

    assert len(o.tasks) == 1
    assert o.tasks[0].status == "completed"


def test_orchestrator_cycle_no_tasks_returns_early() -> None:
    o = Orchestrator(goal="test")
    plan = Plan(tasks=[], rationale="done")
    o.planner = MagicMock()
    o.planner.plan = AsyncMock(return_value=plan)
    o.client = MagicMock()
    o.client.settings = make_settings()
    asyncio.run(o._cycle())
    assert o.tasks == []


def test_orchestrator_cycle_test_failure_updates_goal() -> None:
    o = Orchestrator(goal="original")
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="because",
    )
    o.planner = MagicMock()
    o.planner.plan = AsyncMock(return_value=plan)
    o.client = MagicMock()
    o.client.settings = make_settings()

    import furrow.core.orchestrator as orch_mod
    original_worker = orch_mod.WorkerAgent
    original_tester = orch_mod.TesterAgent
    orch_mod.WorkerAgent = MagicMock()
    orch_mod.WorkerAgent.return_value.run = AsyncMock(return_value="done")

    test_result = TestResult(passed=False, summary="fail", failures=["error1", "error2"])
    orch_mod.TesterAgent = MagicMock()
    orch_mod.TesterAgent.return_value.run = AsyncMock(return_value=test_result)

    try:
        asyncio.run(o._cycle())
    finally:
        orch_mod.WorkerAgent = original_worker
        orch_mod.TesterAgent = original_tester

    assert o.goal.startswith("Fix failing tests:")
    assert "error1" in o.goal
    assert "error2" in o.goal


def test_orchestrator_cycle_test_passes_keeps_goal() -> None:
    o = Orchestrator(goal="original")
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="because",
    )
    o.planner = MagicMock()
    o.planner.plan = AsyncMock(return_value=plan)
    o.client = MagicMock()
    o.client.settings = make_settings()

    import furrow.core.orchestrator as orch_mod
    original_worker = orch_mod.WorkerAgent
    original_tester = orch_mod.TesterAgent
    orch_mod.WorkerAgent = MagicMock()
    orch_mod.WorkerAgent.return_value.run = AsyncMock(return_value="done")

    test_result = TestResult(passed=True, summary="ok", failures=[])
    orch_mod.TesterAgent = MagicMock()
    orch_mod.TesterAgent.return_value.run = AsyncMock(return_value=test_result)

    try:
        asyncio.run(o._cycle())
    finally:
        orch_mod.WorkerAgent = original_worker
        orch_mod.TesterAgent = original_tester

    assert o.goal == "original"


def test_orchestrator_cycle_task_exception_marks_failed() -> None:
    o = Orchestrator(goal="test")
    plan = Plan(
        tasks=[TaskModel(id="1", description="do thing")],
        rationale="because",
    )
    o.planner = MagicMock()
    o.planner.plan = AsyncMock(return_value=plan)
    o.client = MagicMock()
    o.client.settings = make_settings()

    import furrow.core.orchestrator as orch_mod
    original_worker = orch_mod.WorkerAgent
    original_tester = orch_mod.TesterAgent
    orch_mod.WorkerAgent = MagicMock()
    orch_mod.WorkerAgent.return_value.run = AsyncMock(side_effect=RuntimeError("boom"))
    orch_mod.TesterAgent = MagicMock()
    orch_mod.TesterAgent.return_value.run = AsyncMock(return_value=TestResult(passed=True, summary="ok", failures=[]))

    try:
        asyncio.run(o._cycle())
    finally:
        orch_mod.WorkerAgent = original_worker
        orch_mod.TesterAgent = original_tester

    assert o.tasks[0].status == "failed"
    assert "boom" in o.tasks[0].result