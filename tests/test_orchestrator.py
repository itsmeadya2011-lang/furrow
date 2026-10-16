from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator


def make_task(task_id: str, status: str = "completed") -> TaskModel:
    return TaskModel(id=task_id, description=f"task {task_id}", status=status)


def test_is_done_when_all_tasks_completed():
    orch = Orchestrator(goal="x")
    orch.tasks = [make_task("1"), make_task("2")]
    assert orch._is_done() is True


def test_is_done_false_when_no_tasks():
    orch = Orchestrator(goal="x")
    assert orch._is_done() is False


def test_is_done_false_when_any_task_failed():
    orch = Orchestrator(goal="x")
    orch.tasks = [make_task("1"), make_task("2", status="failed")]
    assert orch._is_done() is False


def test_get_tasks_returns_tracked_tasks():
    orch = Orchestrator(goal="x")
    task = make_task("1")
    orch.tasks = [task]
    assert orch._get_tasks() == [task]
