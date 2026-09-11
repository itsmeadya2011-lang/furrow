import pytest
from furrow.config import (
    Plan,
    TaskModel,
    TestResult,
    TaskStatus,
    validate_task_graph,
    transition_start,
    transition_complete,
    transition_fail,
    transition_block,
)


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


# Graph validation tests


def test_validate_task_graph_duplicate_ids():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="1", description="task 2"),
    ]
    with pytest.raises(ValueError, match="duplicate IDs"):
        validate_task_graph(tasks)


def test_validate_task_graph_unknown_dependency():
    tasks = [
        TaskModel(id="1", description="task 1", dependencies=["2"]),
    ]
    with pytest.raises(ValueError, match="unknown task"):
        validate_task_graph(tasks)


def test_validate_task_graph_self_dependency():
    tasks = [
        TaskModel(id="1", description="task 1", dependencies=["1"]),
    ]
    with pytest.raises(ValueError, match="depends on itself"):
        validate_task_graph(tasks)


def test_validate_task_graph_cycle():
    tasks = [
        TaskModel(id="1", description="task 1", dependencies=["2"]),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
    ]
    with pytest.raises(ValueError, match="cycle"):
        validate_task_graph(tasks)


def test_validate_task_graph_valid():
    tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2", dependencies=["1"]),
        TaskModel(id="3", description="task 3", dependencies=["1", "2"]),
    ]
    validate_task_graph(tasks)  # Should not raise


# TaskStatus transition tests


def test_transition_start():
    task = TaskModel(id="1", description="test", status=TaskStatus.PENDING)
    result = transition_start(task)
    assert result.status == TaskStatus.RUNNING
    assert result.error is None
    assert result is task  # Mutates in place


def test_transition_complete():
    task = TaskModel(id="1", description="test", status=TaskStatus.RUNNING, error="old error")
    result = transition_complete(task, result="done")
    assert result.status == TaskStatus.COMPLETED
    assert result.result == "done"
    assert result.error is None
    assert result is task


def test_transition_fail():
    task = TaskModel(id="1", description="test", status=TaskStatus.RUNNING)
    result = transition_fail(task, error="something broke")
    assert result.status == TaskStatus.FAILED
    assert result.error == "something broke"
    assert result is task


def test_transition_block():
    task = TaskModel(id="1", description="test", status=TaskStatus.PENDING)
    result = transition_block(task, reason="waiting on dep")
    assert result.status == TaskStatus.BLOCKED
    assert result.error == "waiting on dep"
    assert result is task
