"""Tests for furrow.core.orchestrator: state, cycles, broadcast, and run loop."""

import json
from unittest.mock import AsyncMock

import pytest

from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def orch_settings(monkeypatch, tmp_path):
    custom = Settings(
        workspace=tmp_path,
        provider=Provider.ANTHROPIC,
        anthropic_api_key="sk-test",
    )
    monkeypatch.setattr("furrow.core.orchestrator.settings", custom)
    return custom


def test_state_file_property(orch_settings, tmp_path):
    orch = Orchestrator(goal="g")
    assert orch.state_file == tmp_path / ".furrow" / "state.json"


def test_init_stores_goal(orch_settings):
    orch = Orchestrator(goal="build a thing")
    assert orch.goal == "build a thing"
    assert orch.cycles == 0
    assert orch._tasks == []
    assert orch._stopped is False


def test_stop_sets_stopped_flag(orch_settings):
    orch = Orchestrator(goal="g")
    assert orch._stopped is False
    orch.stop()
    assert orch._stopped is True


def test_get_tasks_returns_internal_tasks(orch_settings):
    orch = Orchestrator(goal="g")
    tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
    orch._tasks = tasks
    assert orch._get_tasks() == tasks
    assert orch._get_tasks() is orch._tasks


def test_broadcast_calls_progress_callback_when_set(orch_settings):
    received = []
    orch = Orchestrator(goal="g", progress_callback=lambda m: received.append(m))
    orch._broadcast("hello")
    assert received == ["hello"]


def test_broadcast_no_op_when_callback_none(orch_settings):
    orch = Orchestrator(goal="g")
    orch.progress_callback = None
    orch._broadcast("ignored")


def test_is_done_true_when_max_cycles_reached(orch_settings):
    orch_settings.max_cycles = 1
    orch = Orchestrator(goal="g")
    orch.cycles = 1
    assert orch._is_done() is True


def test_is_done_true_when_max_cycles_reached_even_if_tests_failed(orch_settings):
    orch_settings.max_cycles = 1
    orch = Orchestrator(goal="g")
    orch.cycles = 1
    orch._last_test_result = TestResult(passed=False, summary="bad", failures=["x"])
    assert orch._is_done() is True


def test_is_done_true_when_all_tasks_completed(orch_settings):
    orch_settings.max_cycles = 0
    orch = Orchestrator(goal="g")
    orch._tasks = [TaskModel(id="1", description="task", status="completed")]
    assert orch._is_done() is True


def test_is_done_false_when_tasks_still_pending(orch_settings):
    orch_settings.max_cycles = 0
    orch = Orchestrator(goal="g")
    orch._tasks = [TaskModel(id="1", description="task", status="pending")]
    assert orch._is_done() is False


def test_is_done_false_when_any_task_failed(orch_settings):
    orch_settings.max_cycles = 0
    orch = Orchestrator(goal="g")
    orch._tasks = [TaskModel(id="1", description="task", status="failed")]
    assert orch._is_done() is False


def test_is_done_false_when_last_test_result_failed(orch_settings):
    orch_settings.max_cycles = 0
    orch = Orchestrator(goal="g")
    orch._tasks = [TaskModel(id="1", description="task", status="completed")]
    orch._last_test_result = TestResult(passed=False, summary="bad", failures=["x"])
    assert orch._is_done() is False


def test_is_done_true_when_no_tasks_and_no_test_result(orch_settings):
    orch_settings.max_cycles = 0
    orch = Orchestrator(goal="g")
    assert orch._is_done() is True


def test_save_state_writes_json(orch_settings, tmp_path):
    orch = Orchestrator(goal="my goal")
    orch.cycles = 3
    orch._tasks = [TaskModel(id="1", description="d", status="completed", result="done")]
    orch._save_state()
    state_path = tmp_path / ".furrow" / "state.json"
    assert state_path.exists()
    data = json.loads(state_path.read_text())
    assert data["goal"] == "my goal"
    assert data["cycles"] == 3
    assert len(data["tasks"]) == 1
    assert data["tasks"][0]["id"] == "1"
    assert data["tasks"][0]["status"] == "completed"


def test_load_state_reads_json(orch_settings, tmp_path):
    state = {
        "goal": "loaded goal",
        "cycles": 7,
        "tasks": [{"id": "t1", "description": "desc", "status": "pending"}],
    }
    state_path = tmp_path / ".furrow" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(state))
    orch = Orchestrator(goal="original goal")
    assert orch.goal == "loaded goal"
    assert orch.cycles == 7
    assert len(orch._tasks) == 1
    assert orch._tasks[0].id == "t1"


def test_load_state_missing_file_graceful(orch_settings, tmp_path):
    orch = Orchestrator(goal="g")
    assert orch.goal == "g"
    assert orch.cycles == 0
    assert orch._tasks == []


def test_load_state_corrupted_json_graceful(orch_settings, tmp_path):
    state_path = tmp_path / ".furrow" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("{not valid json,,")
    orch = Orchestrator(goal="g")
    assert orch.goal == "g"
    assert orch.cycles == 0
    assert orch._tasks == []


async def test_run_completes_when_max_cycles_reached(orch_settings, monkeypatch):
    orch_settings.max_cycles = 1
    orch = Orchestrator(goal="g")
    plan = Plan(tasks=[], rationale="done")
    orch.planner.plan = AsyncMock(return_value=plan)
    orch.tester.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )
    await orch.run()
    assert orch.cycles == 1
    orch.planner.plan.assert_awaited()
    orch.tester.run.assert_awaited()


async def test_run_stops_immediately_when_already_stopped(orch_settings):
    orch_settings.max_cycles = 5
    orch = Orchestrator(goal="g")
    orch.stop()
    plan = Plan(tasks=[], rationale="nope")
    orch.planner.plan = AsyncMock(return_value=plan)
    orch.tester.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )
    await orch.run()
    assert orch.cycles == 0
    orch.planner.plan.assert_not_awaited()
    orch.tester.run.assert_not_awaited()


async def test_run_cycle_with_executed_tasks(orch_settings, monkeypatch):
    orch_settings.max_cycles = 1
    orch = Orchestrator(goal="g")
    plan = Plan(tasks=[TaskModel(id="1", description="work", files=[])], rationale="do it")
    orch.planner.plan = AsyncMock(return_value=plan)
    orch.tester.run = AsyncMock(
        return_value=TestResult(passed=True, summary="ok", failures=[])
    )

    class FakeWorkerAgent:
        def __init__(self, task, client):
            self.task = task

        async def run(self):
            return "done"

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorkerAgent)
    await orch.run()
    assert orch.cycles == 1
    assert orch._tasks[0].status == "completed"
    assert orch._tasks[0].result == "done"
