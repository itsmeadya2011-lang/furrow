import pytest
from furrow.config import Plan, TaskModel, TestResult


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_orchestrator_is_done_empty_plan():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="noop")
    orch.last_plan = Plan(tasks=[], rationale="all done")
    assert orch._is_done() is True


def test_orchestrator_is_done_incomplete_tasks():
    from furrow.core.orchestrator import Orchestrator

    t = TaskModel(id="1", description="x", status="pending")
    orch = Orchestrator(goal="noop")
    orch.last_plan = Plan(tasks=[t], rationale="keep going")
    assert orch._is_done() is False


def test_orchestrator_is_done_all_completed_with_rationale():
    from furrow.core.orchestrator import Orchestrator

    t = TaskModel(id="1", description="x", status="completed")
    orch = Orchestrator(goal="noop")
    orch.last_plan = Plan(tasks=[t], rationale="Goal COMPLETE")
    orch.completed_task_ids.add("1")
    assert orch._is_done() is True


def test_orchestrator_tracks_failed_task_ids():
    from furrow.core.orchestrator import Orchestrator

    t1 = TaskModel(id="1", description="x", status="completed")
    t2 = TaskModel(id="2", description="y", status="failed")
    orch = Orchestrator(goal="noop")
    orch.last_plan = Plan(tasks=[t1, t2], rationale="DONE")
    orch.completed_task_ids.add("1")
    orch.failed_task_ids.add("2")
    assert orch._is_done() is False
