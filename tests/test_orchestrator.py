from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import LogEvent, Orchestrator, OrchestratorStop
from furrow.core.state import State


def _task(tid: str, status: str, description: str = "d") -> TaskModel:
    return TaskModel(id=tid, description=description, status=status)


# --- _is_done logic -------------------------------------------------------
def test_is_done_empty_no_result():
    o = Orchestrator(goal="x")
    assert o._is_done() is True


def test_is_done_empty_but_tests_failed():
    o = Orchestrator(goal="x")
    o._last_test_result = TestResult(passed=False, summary="fail", failures=["f"])
    assert o._is_done() is False


def test_is_done_all_completed_no_tests():
    o = Orchestrator(goal="x")
    o._tasks = [_task("1", "completed"), _task("2", "completed")]
    assert o._is_done() is True


def test_is_done_all_completed_tests_failed():
    """Bug fix: completed tasks but failing tests must NOT be done."""
    o = Orchestrator(goal="x")
    o._tasks = [_task("1", "completed")]
    o._last_test_result = TestResult(passed=False, summary="fail", failures=["f"])
    assert o._is_done() is False


def test_is_done_all_completed_tests_passed():
    o = Orchestrator(goal="x")
    o._tasks = [_task("1", "completed")]
    o._last_test_result = TestResult(passed=True, summary="ok")
    assert o._is_done() is True


def test_is_done_a_task_failed():
    o = Orchestrator(goal="x")
    o._tasks = [_task("1", "failed")]
    o._last_test_result = TestResult(passed=True, summary="ok")
    assert o._is_done() is False


def test_is_done_incomplete_tasks():
    o = Orchestrator(goal="x")
    o._tasks = [_task("1", "completed"), _task("2", "pending")]
    o._last_test_result = TestResult(passed=True, summary="ok")
    assert o._is_done() is False


def test_get_tasks_returns_current_tasks():
    o = Orchestrator(goal="x")
    o._tasks = [_task("1", "completed")]
    assert o._get_tasks() == o._tasks


# --- run loop with mocked agents -----------------------------------------
async def _run_with_mocks(monkeypatch, orchestrator_module, plan, test_result,
                          settings_kwargs=None):
    s = Settings(**(settings_kwargs or {}))
    monkeypatch.setattr(orchestrator_module.PlannerAgent, "plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(orchestrator_module.WorkerAgent, "run", AsyncMock(return_value="done"))
    monkeypatch.setattr(orchestrator_module.TesterAgent, "run", AsyncMock(return_value=test_result))
    o = Orchestrator(goal="g", settings=s)
    await o.run()
    return o


@pytest.mark.asyncio
async def test_run_stops_on_max_cycles(monkeypatch):
    import furrow.core.orchestrator as orch_mod

    plan = Plan(tasks=[TaskModel(id="1", description="t", status="pending")], rationale="r")
    failing = TestResult(passed=False, summary="fail", failures=["broken thing"])
    o = await _run_with_mocks(monkeypatch, orch_mod, plan, failing, {"max_cycles": 1})
    # Two cycle iterations attempted: cycle 1 runs (fails tests), cycle 2 hits max_cycles.
    assert o.cycles == 2


@pytest.mark.asyncio
async def test_run_completes_when_tests_pass(monkeypatch, tmp_path):
    import furrow.core.orchestrator as orch_mod

    plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="r")
    passing = TestResult(passed=True, summary="all good")
    # Persist so we also verify the state file is written.
    settings = Settings(state_file=str(tmp_path / "st.json"))
    monkeypatch.setattr(orch_mod.PlannerAgent, "plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(orch_mod.WorkerAgent, "run", AsyncMock(return_value="done"))
    monkeypatch.setattr(orch_mod.TesterAgent, "run", AsyncMock(return_value=passing))
    o = Orchestrator(goal="g", settings=settings, persist=True)
    await o.run()
    assert o.cycles == 1
    assert (tmp_path / "st.json").exists()


@pytest.mark.asyncio
async def test_run_emits_to_listeners(monkeypatch, tmp_path):
    import furrow.core.orchestrator as orch_mod

    plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="r")
    passing = TestResult(passed=True, summary="ok")
    received = []

    async def listener(event: LogEvent):
        received.append(event.message)

    settings = Settings(state_file=str(tmp_path / "st.json"))
    monkeypatch.setattr(orch_mod.PlannerAgent, "plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(orch_mod.WorkerAgent, "run", AsyncMock(return_value="done"))
    monkeypatch.setattr(orch_mod.TesterAgent, "run", AsyncMock(return_value=passing))
    o = Orchestrator(
        goal="g", settings=settings, persist=True, listeners=[listener]
    )
    await o.run()
    assert any("Plan" in m for m in received)
    assert any("Tests passed" in m for m in received)


@pytest.mark.asyncio
async def test_listener_stop_halts_loop(monkeypatch):
    import furrow.core.orchestrator as orch_mod

    plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="r")
    failing = TestResult(passed=False, summary="fail", failures=["f"])
    settings = Settings(max_cycles=0)  # unlimited; should stop via listener

    async def stop_listener(event: LogEvent):
        if "Will attempt fix" in event.message:
            raise OrchestratorStop()

    monkeypatch.setattr(orch_mod.PlannerAgent, "plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(orch_mod.WorkerAgent, "run", AsyncMock(return_value="done"))
    monkeypatch.setattr(orch_mod.TesterAgent, "run", AsyncMock(return_value=failing))
    o = Orchestrator(goal="g", settings=settings, listeners=[stop_listener])
    await o.run()  # should not hang
    assert o.cycles >= 1


@pytest.mark.asyncio
async def test_run_resumes_from_state(monkeypatch, tmp_path):
    import furrow.core.orchestrator as orch_mod

    # Pre-write a state file with an existing goal matching the new one.
    state_file = tmp_path / "st.json"
    existing = State(goal="g", cycle=2, tasks=[], test_result=None)
    mgr = orch_mod.StateManager(state_file)
    await mgr.save(existing)

    plan = Plan(tasks=[TaskModel(id="1", description="t")], rationale="r")
    passing = TestResult(passed=True, summary="ok")
    settings = Settings(state_file=str(state_file))
    monkeypatch.setattr(orch_mod.PlannerAgent, "plan", AsyncMock(return_value=plan))
    monkeypatch.setattr(orch_mod.WorkerAgent, "run", AsyncMock(return_value="done"))
    monkeypatch.setattr(orch_mod.TesterAgent, "run", AsyncMock(return_value=passing))
    o = Orchestrator(goal="g", settings=settings, persist=True)
    await o.run()
    # Resumed at cycle 2, then one more cycle runs -> cycle 3, then done.
    assert o.cycles >= 2
