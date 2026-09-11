from __future__ import annotations

from unittest.mock import MagicMock

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


# ---------------------------------------------------------------------------
# Existing tests
# ---------------------------------------------------------------------------


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


def _make_orchestrator(**kwargs) -> Orchestrator:
    client = MagicMock()
    client.settings.max_cycles = kwargs.pop("max_cycles", 0)
    return Orchestrator(goal="test goal", client=client, **kwargs)


def test_orchestrator_init():
    client = MagicMock()
    client.settings.max_cycles = 0
    o = Orchestrator(goal="do thing", client=client)

    assert hasattr(o, "_tasks")
    assert hasattr(o, "max_cycles")
    assert hasattr(o, "_last_test_result")
    assert o._tasks == []
    assert o._last_test_result is None


def test_orchestrator_get_tasks_empty():
    o = _make_orchestrator()
    assert o._get_tasks() == []


def test_orchestrator_is_done_no_tasks():
    o = _make_orchestrator()
    assert o._is_done() is True


def test_orchestrator_is_done_all_completed():
    o = _make_orchestrator()
    o._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert o._is_done() is True


def test_orchestrator_is_done_with_failed_task():
    o = _make_orchestrator()
    o._tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert o._is_done() is False


def test_orchestrator_is_done_with_failing_tests():
    o = _make_orchestrator()
    o._tasks = [
        TaskModel(id="1", description="a", status="completed"),
    ]
    o._last_test_result = TestResult(passed=False, summary="fail", failures=["bad"])
    assert o._is_done() is False


def test_orchestrator_is_done_with_passing_tests():
    o = _make_orchestrator()
    o._tasks = [
        TaskModel(id="1", description="a", status="completed"),
    ]
    o._last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert o._is_done() is True


def test_orchestrator_max_cycles():
    o = _make_orchestrator(max_cycles=3)
    assert o.max_cycles == 3


def test_orchestrator_console_injection():
    console = MagicMock()
    o = _make_orchestrator(console=console)
    assert o.console is console


def test_task_model_defaults():
    t = TaskModel(id="1", description="do thing")
    assert t.status == "pending"
    assert t.files == []
    assert t.dependencies == []
    assert t.result is None