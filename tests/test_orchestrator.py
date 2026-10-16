from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator


def _task(task_id: str, status: str = "pending") -> TaskModel:
    return TaskModel(id=task_id, description=f"task {task_id}", status=status)


def test_get_tasks_returns_stored_tasks_not_empty() -> None:
    orch = Orchestrator(goal="do thing")
    orch.tasks = [_task("1"), _task("2")]
    assert orch._get_tasks() == orch.tasks
    assert len(orch._get_tasks()) == 2


def test_is_done_true_when_no_tasks() -> None:
    orch = Orchestrator(goal="do thing")
    assert orch._is_done() is True


def test_is_done_true_when_all_completed() -> None:
    orch = Orchestrator(goal="do thing")
    orch.tasks = [_task("1", "completed"), _task("2", "completed")]
    orch.last_test_passed = True
    assert orch._is_done() is True


def test_is_done_false_when_any_failed() -> None:
    orch = Orchestrator(goal="do thing")
    orch.tasks = [_task("1", "completed"), _task("2", "failed")]
    assert orch._is_done() is False


def test_is_done_false_when_tests_failed_even_if_tasks_completed() -> None:
    orch = Orchestrator(goal="do thing")
    orch.tasks = [_task("1", "completed"), _task("2", "completed")]
    orch.last_test_passed = False
    assert orch._is_done() is False
