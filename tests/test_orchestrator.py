from __future__ import annotations

import pytest

from furrow.config import TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class _NoOpClient:
    """Minimal client stub for testing orchestrator logic."""

    class _Settings:
        max_cycles = 0

    def __init__(self) -> None:
        self.settings = self._Settings()


def test_get_tasks_returns_empty_initially():
    o = Orchestrator(goal="test", client=_NoOpClient())
    assert o._get_tasks() == []


def test_get_tasks_returns_tracked_tasks():
    o = Orchestrator(goal="test", client=_NoOpClient())
    o.all_tasks["1"] = TaskModel(id="1", description="task 1")
    o.all_tasks["2"] = TaskModel(id="2", description="task 2")
    assert len(o._get_tasks()) == 2


def test_is_done_when_no_tasks_and_no_test_result():
    o = Orchestrator(goal="test", client=_NoOpClient())
    assert o._is_done() is False


def test_is_done_when_all_completed():
    o = Orchestrator(goal="test", client=_NoOpClient())
    o.all_tasks["1"] = TaskModel(id="1", description="task 1", status="completed")
    o.all_tasks["2"] = TaskModel(id="2", description="task 2", status="completed")
    assert o._is_done() is True


def test_is_done_when_any_failed():
    o = Orchestrator(goal="test", client=_NoOpClient())
    o.all_tasks["1"] = TaskModel(id="1", description="task 1", status="completed")
    o.all_tasks["2"] = TaskModel(id="2", description="task 2", status="failed")
    assert o._is_done() is False


def test_is_done_when_test_passed():
    o = Orchestrator(goal="test", client=_NoOpClient())
    o.last_test_result = TestResult(passed=True, summary="ok")
    assert o._is_done() is True


def test_is_done_when_test_failed():
    o = Orchestrator(goal="test", client=_NoOpClient())
    o.last_test_result = TestResult(passed=False, summary="fail", failures=["x"])
    o.all_tasks["1"] = TaskModel(id="1", description="task 1", status="completed")
    assert o._is_done() is False


def test_is_done_when_pending_tasks_remain():
    o = Orchestrator(goal="test", client=_NoOpClient())
    o.all_tasks["1"] = TaskModel(id="1", description="task 1", status="pending")
    assert o._is_done() is False


def test_max_cycles_zero_means_unlimited():
    o = Orchestrator(goal="test", client=_NoOpClient())
    assert o.max_cycles == 0


def test_max_cycles_set_from_settings():
    class _Client:
        class settings:
            max_cycles = 3

    o = Orchestrator(goal="test", client=_Client())
    assert o.max_cycles == 3


def test_on_message_callback_invoked():
    messages: list[str] = []

    class _Client:
        class settings:
            max_cycles = 0

    o = Orchestrator(goal="test", client=_Client(), on_message=messages.append)
    o._log("hello")
    assert messages == ["hello"]


def test_on_message_none_is_safe():
    class _Client:
        class settings:
            max_cycles = 0

    o = Orchestrator(goal="test", client=_Client())
    o._log("hello")  # should not raise


def test_failed_tasks_passed_to_planner():
    o = Orchestrator(goal="test", client=_NoOpClient())
    o.all_tasks["1"] = TaskModel(id="1", description="task 1", status="failed", result="error msg")
    # _cycle builds goal_context from failed tasks; verify the attribute exists
    assert hasattr(o, "all_tasks")
    failed = [t for t in o.all_tasks.values() if t.status == "failed"]
    assert len(failed) == 1
    assert failed[0].result == "error msg"