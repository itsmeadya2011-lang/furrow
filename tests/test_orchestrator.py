from __future__ import annotations

import asyncio

from furrow.config import Plan, TaskModel, TestResult
from furrow.core import orchestrator as orch_mod
from furrow.core.orchestrator import Orchestrator


def _make():
    return Orchestrator(goal="improve furrow")


# --- _get_tasks / _is_done ---

def test_get_tasks_returns_persisted_tasks():
    orch = _make()
    t = TaskModel(id="1", description="do thing")
    orch._tasks = [t]
    assert orch._get_tasks() == [t]


def test_is_done_empty_plan_is_done():
    orch = _make()
    orch._tasks = []
    assert orch._is_done() is True


def test_is_done_all_completed_and_tests_passed():
    orch = _make()
    orch._tasks = [TaskModel(id="1", description="x", status="completed")]
    orch._last_test = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_is_done_tests_failing_continues():
    orch = _make()
    orch._tasks = [TaskModel(id="1", description="x", status="completed")]
    orch._last_test = TestResult(passed=False, summary="bad", failures=["f"])
    assert orch._is_done() is False


def test_is_done_any_failed_continues():
    orch = _make()
    orch._tasks = [TaskModel(id="1", description="x", status="failed")]
    assert orch._is_done() is False


def test_is_done_not_all_completed_continues():
    orch = _make()
    orch._tasks = [TaskModel(id="1", description="x", status="pending")]
    assert orch._is_done() is False


def test_is_done_no_test_result_yet_continues():
    orch = _make()
    orch._tasks = [TaskModel(id="1", description="x", status="completed")]
    assert orch._is_done() is False


# --- max_cycles termination ---

def test_max_cycles_terminates_loop(monkeypatch):
    orch = _make()
    orch.settings.max_cycles = 2

    async def fake_cycle():
        orch._tasks = [TaskModel(id="1", description="x", status="pending")]
        orch._last_test = None

    monkeypatch.setattr(orch, "_cycle", fake_cycle)
    asyncio.run(orch.run())
    assert orch.cycles == 2


def test_unlimited_cycles_when_max_zero(monkeypatch):
    orch = _make()
    orch.settings.max_cycles = 0

    async def fake_cycle():
        orch._tasks = []
        orch._last_test = None

    monkeypatch.setattr(orch, "_cycle", fake_cycle)
    asyncio.run(orch.run())
    assert orch.cycles == 1


# --- concurrency cap ---

def test_max_parallel_tasks_caps_concurrency(monkeypatch):
    orch = _make()
    n = 6
    tasks = [TaskModel(id=str(i), description=f"t{i}") for i in range(1, n + 1)]
    plan = Plan(tasks=tasks, rationale="r")

    async def fake_plan(goal):
        return plan

    monkeypatch.setattr(orch.planner, "plan", fake_plan)

    state = {"current": 0, "max_seen": 0}

    async def fake_worker_run(self):
        state["current"] += 1
        state["max_seen"] = max(state["max_seen"], state["current"])
        await asyncio.sleep(0.02)
        state["current"] -= 1

    monkeypatch.setattr(orch_mod.WorkerAgent, "run", fake_worker_run)

    async def fake_tester_run(self, goal, tsks):
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr(orch_mod.TesterAgent, "run", fake_tester_run)

    orch.settings.max_parallel_tasks = 2
    asyncio.run(orch.run())

    assert orch.cycles == 1
    assert state["max_seen"] <= 2
    assert state["max_seen"] >= 2
