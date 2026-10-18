import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _make_orch():
    client = MagicMock()
    settings = MagicMock()
    settings.max_cycles = 1
    client.settings = settings
    orch = Orchestrator(goal="g", client=client)
    return orch


def test_is_done_no_tasks():
    orch = _make_orch()
    assert orch._is_done() is True


def test_is_done_failed():
    orch = _make_orch()
    orch._plan = Plan(tasks=[TaskModel(id="1", description="d", status="failed")], rationale="r")
    assert orch._is_done() is False


def test_is_done_all_completed():
    orch = _make_orch()
    orch._plan = Plan(tasks=[TaskModel(id="1", description="d", status="completed")], rationale="r")
    assert orch._is_done() is True


def test_is_done_pending():
    orch = _make_orch()
    orch._plan = Plan(tasks=[TaskModel(id="1", description="d", status="pending")], rationale="r")
    assert orch._is_done() is False


def test_record_completed_dedupe():
    orch = _make_orch()
    t1 = TaskModel(id="1", description="d", status="completed", result="r1")
    orch._record_completed(t1)
    t2 = TaskModel(id="1", description="d", status="completed", result="r2")
    orch._record_completed(t2)
    assert len(orch._completed_tasks) == 1
    assert orch._completed_tasks[0].result == "r2"


def test_record_completed_append_new():
    orch = _make_orch()
    orch._record_completed(TaskModel(id="1", description="d", status="completed"))
    orch._record_completed(TaskModel(id="2", description="d2", status="completed"))
    assert len(orch._completed_tasks) == 2


def test_get_tasks_union():
    orch = _make_orch()
    orch._plan = Plan(tasks=[TaskModel(id="1", description="d", status="completed")], rationale="r")
    orch._completed_tasks = [TaskModel(id="1", description="d", status="completed"), TaskModel(id="2", description="d2", status="completed")]
    tasks = orch._get_tasks()
    ids = [t.id for t in tasks]
    assert ids == ["1", "2"]


def test_run_returns_str():
    orch = _make_orch()
    orch.settings.max_cycles = 1
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    with patch("furrow.core.orchestrator.PlannerAgent.plan", new=AsyncMock(return_value=plan)):
        with patch("furrow.core.orchestrator.WorkerAgent.run", new=AsyncMock(side_effect=Exception("boom"))):
            with patch("furrow.core.orchestrator.TesterAgent.run", new=AsyncMock(return_value=TestResult(passed=False, summary="fail", failures=["x"]))):
                result = asyncio.run(orch.run())
    assert isinstance(result, str)


def test_max_cycles():
    orch = _make_orch()
    orch.settings.max_cycles = 2
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    with patch("furrow.core.orchestrator.PlannerAgent.plan", new=AsyncMock(return_value=plan)):
        with patch("furrow.core.orchestrator.WorkerAgent.run", new=AsyncMock(side_effect=Exception("boom"))):
            with patch("furrow.core.orchestrator.TesterAgent.run", new=AsyncMock(return_value=TestResult(passed=False, summary="fail", failures=["x"]))):
                result = asyncio.run(orch.run())
    assert isinstance(result, str)
    assert "max_cycles" in result


def test_output_callback_sync():
    calls = []
    client = MagicMock()
    settings = MagicMock()
    settings.max_cycles = 1
    client.settings = settings
    orch = Orchestrator(goal="g", client=client, output_callback=lambda m: calls.append(m))
    plan = Plan(tasks=[], rationale="ok")
    with patch("furrow.core.orchestrator.PlannerAgent.plan", new=AsyncMock(return_value=plan)):
        asyncio.run(orch.run())
    assert len(calls) > 0


def test_output_callback_async():
    calls = []

    async def cb(m):
        calls.append(m)

    client = MagicMock()
    settings = MagicMock()
    settings.max_cycles = 1
    client.settings = settings
    orch = Orchestrator(goal="g", client=client, output_callback=cb)
    plan = Plan(tasks=[], rationale="ok")
    with patch("furrow.core.orchestrator.PlannerAgent.plan", new=AsyncMock(return_value=plan)):
        asyncio.run(orch.run())
    assert len(calls) > 0
