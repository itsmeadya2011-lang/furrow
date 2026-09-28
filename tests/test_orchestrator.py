from unittest.mock import MagicMock

from furrow.config import TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _task(tid: str = "1", status: str = "completed") -> TaskModel:
    t = TaskModel(id=tid, description="example task")
    t.status = status
    return t


def _orch() -> Orchestrator:
    return Orchestrator(goal="demo", client=MagicMock())


def test_get_tasks_returns_stored_tasks():
    """Regression: _get_tasks must return plan tasks, not an empty list."""
    orch = _orch()
    orch.tasks = [_task("1"), _task("2")]
    assert orch._get_tasks() is orch.tasks
    assert len(orch._get_tasks()) == 2


def test_is_done_when_all_tasks_complete_and_tests_pass():
    orch = _orch()
    orch.tasks = [_task("1"), _task("2")]
    orch.last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is True


def test_not_done_when_tests_fail():
    orch = _orch()
    orch.tasks = [_task("1"), _task("2")]
    orch.last_test_result = TestResult(passed=False, summary="bad", failures=["f1"])
    assert orch._is_done() is False


def test_not_done_when_any_task_failed():
    orch = _orch()
    orch.tasks = [_task("1", "failed"), _task("2")]
    orch.last_test_result = TestResult(passed=True, summary="ok")
    assert orch._is_done() is False


def test_is_done_when_no_tasks():
    orch = _orch()
    orch.tasks = []
    orch.last_test_result = None
    assert orch._is_done() is True
