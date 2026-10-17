from furrow.config import TaskModel
from furrow.core.orchestrator import Orchestrator


def make_task(tid: str, status: str = "pending") -> TaskModel:
    return TaskModel(id=tid, description=f"task {tid}", status=status)


def test_is_done_empty_tasks():
    o = Orchestrator(goal="test")
    assert o._is_done() is False


def test_is_done_all_completed():
    o = Orchestrator(goal="test")
    o.all_tasks = [make_task("1", "completed"), make_task("2", "completed")]
    assert o._is_done() is True


def test_is_done_one_failed():
    o = Orchestrator(goal="test")
    o.all_tasks = [make_task("1", "completed"), make_task("2", "failed")]
    assert o._is_done() is False


def test_is_done_one_pending():
    o = Orchestrator(goal="test")
    o.all_tasks = [make_task("1", "completed"), make_task("2", "pending")]
    assert o._is_done() is False


def test_is_done_max_cycles_reached():
    o = Orchestrator(goal="test", max_cycles=3)
    o.cycles = 3
    assert o._is_done() is True


def test_is_done_max_cycles_not_reached():
    o = Orchestrator(goal="test", max_cycles=3)
    o.cycles = 2
    assert o._is_done() is False


def test_is_done_max_cycles_unlimited():
    o = Orchestrator(goal="test", max_cycles=0)
    o.cycles = 1000
    assert o._is_done() is False


def test_max_parallel_defaults_to_settings():
    from furrow.config import settings
    o = Orchestrator(goal="test")
    assert o.max_parallel == settings.max_parallel_tasks


def test_all_tasks_tracked_across_cycles():
    o = Orchestrator(goal="test")
    o.all_tasks.append(make_task("1", "completed"))
    o.all_tasks.append(make_task("2", "pending"))
    assert len(o.all_tasks) == 2
    assert o.all_tasks[0].status == "completed"
    assert o.all_tasks[1].status == "pending"
