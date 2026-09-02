from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.core.orchestrator import Orchestrator


def _make_plan(task_ids: list[str], deps: dict[str, list[str]] | None = None) -> Plan:
    deps = deps or {}
    return Plan(
        tasks=[
            TaskModel(id=tid, description=f"task {tid}", dependencies=deps.get(tid, []))
            for tid in task_ids
        ],
        rationale="ok",
    )


@pytest.fixture
def patched_agents() -> tuple[AsyncMock, AsyncMock, AsyncMock]:
    plan = _make_plan(["t1", "t2"])
    planner_mock = AsyncMock(return_value=plan)
    worker_mock = AsyncMock(return_value="ok")
    tester_mock = AsyncMock()
    with (
        patch("furrow.core.orchestrator.PlannerAgent.plan", planner_mock),
        patch("furrow.core.orchestrator.WorkerAgent.run", worker_mock),
        patch("furrow.core.orchestrator.TesterAgent.run", tester_mock),
    ):
        # All worker calls mark tasks completed and tests pass by default.
        async def _tester(goal, tasks):
            for t in tasks:
                t.status = "completed"
            return TestResult(passed=True, summary="ok", failures=[])
        tester_mock.side_effect = _tester
        yield planner_mock, worker_mock, tester_mock


@pytest.mark.asyncio
async def test_run_completes_in_one_cycle(patched_agents) -> None:
    orch = Orchestrator(goal="x")
    await orch.run()
    assert orch.cycles == 1
    assert orch._done_reason == "complete"


@pytest.mark.asyncio
async def test_run_retries_on_test_failure() -> None:
    plan = _make_plan(["t1"])
    planner_mock = AsyncMock(return_value=plan)
    call_count = {"n": 0}

    async def _tester(goal, tasks):
        call_count["n"] += 1
        if call_count["n"] == 1:
            for t in tasks:
                t.status = "completed"
            return TestResult(passed=False, summary="bad", failures=["x"])
        for t in tasks:
            t.status = "completed"
        return TestResult(passed=True, summary="ok", failures=[])

    worker_mock = AsyncMock(return_value="ok")
    tester_mock = AsyncMock(side_effect=_tester)

    with (
        patch("furrow.core.orchestrator.PlannerAgent.plan", planner_mock),
        patch("furrow.core.orchestrator.WorkerAgent.run", worker_mock),
        patch("furrow.core.orchestrator.TesterAgent.run", tester_mock),
    ):
        orch = Orchestrator(goal="x")
        await orch.run()
    assert orch.cycles == 2
    assert orch._done_reason == "complete"


@pytest.mark.asyncio
async def test_max_cycles_enforced(monkeypatch) -> None:
    plan = _make_plan(["t1"])
    planner_mock = AsyncMock(return_value=plan)

    async def _tester(goal, tasks):
        for t in tasks:
            t.status = "completed"
        return TestResult(passed=False, summary="bad", failures=["x"])

    worker_mock = AsyncMock(return_value="ok")
    tester_mock = AsyncMock(side_effect=_tester)

    monkeypatch.setattr(settings, "max_cycles", 2)

    with (
        patch("furrow.core.orchestrator.PlannerAgent.plan", planner_mock),
        patch("furrow.core.orchestrator.WorkerAgent.run", worker_mock),
        patch("furrow.core.orchestrator.TesterAgent.run", tester_mock),
    ):
        orch = Orchestrator(goal="x")
        await orch.run()
    assert orch.cycles == 2
    assert orch._done_reason == "max_cycles"


@pytest.mark.asyncio
async def test_event_callback_receives_events(patched_agents) -> None:
    events: list[tuple[str, dict]] = []

    async def on_event(name: str, data: dict) -> None:
        events.append((name, data))

    orch = Orchestrator(goal="x", on_event=on_event)
    await orch.run()
    names = [n for n, _ in events]
    assert "cycle_start" in names
    assert "done" in names


@pytest.mark.asyncio
async def test_stop_event_halts() -> None:
    stop = asyncio.Event()
    stop.set()  # already set before run starts
    orch = Orchestrator(goal="x", stop_event=stop)
    await orch.run()
    assert orch._done_reason == "stopped"


def test_dependency_waves() -> None:
    plan = _make_plan(
        ["t1", "t2", "t3"],
        deps={"t2": ["t1"], "t3": ["t2"]},
    )
    orch = Orchestrator(goal="x")
    waves = orch._build_waves(plan.tasks)
    assert waves == [{"t1"}, {"t2"}, {"t3"}]


def test_dependency_waves_independent() -> None:
    plan = _make_plan(["a", "b", "c"])
    orch = Orchestrator(goal="x")
    waves = orch._build_waves(plan.tasks)
    assert waves == [{"a", "b", "c"}]