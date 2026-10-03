"""The project's original regression guard, plus core model composition checks.

The two tests below were here first: they pin the shape of a parsed plan and of
a test verdict. They must keep passing unchanged.
"""

from __future__ import annotations

from furrow.config import Plan, TaskModel, TaskStatus, TestResult
from tests.conftest import make_plan, make_task


def test_plan_parse() -> None:
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result() -> None:
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_plan_keeps_the_order_the_planner_produced() -> None:
    plan = make_plan([make_task(id=str(index)) for index in range(5)])
    assert [task.id for task in plan.tasks] == ["0", "1", "2", "3", "4"]


def test_a_fresh_task_starts_pending_and_not_done() -> None:
    task = make_task()
    assert task.status is TaskStatus.PENDING
    assert task.is_done() is False
    assert task.is_terminal() is False


def test_a_completed_task_is_both_done_and_terminal() -> None:
    task = make_task(status=TaskStatus.COMPLETED)
    assert task.is_done() is True
    assert task.is_terminal() is True


def test_plan_survives_a_json_round_trip_with_mixed_task_states() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="done", description="d", status=TaskStatus.COMPLETED, attempts=1),
            TaskModel(id="stopped", description="d", status=TaskStatus.SKIPPED),
        ],
        rationale="mixed",
    )
    restored = Plan.model_validate_json(plan.model_dump_json())
    assert restored.tasks[0].is_done()
    assert restored.tasks[1].is_terminal()
    assert not restored.tasks[1].is_done()
