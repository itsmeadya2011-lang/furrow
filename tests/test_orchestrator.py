"""Tests for the plan/execute/test loop in :mod:`furrow.core.orchestrator`.

The three agents are always replaced with fakes, so nothing here contacts an
LLM. Covered: the loop's termination rules and cycle accounting, the
dependency-level scheduling in ``_schedule_levels``, preservation of the
original goal across a failing cycle, and the ``on_event`` progress stream.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from typing import Any

import pytest

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, TaskModel, TestResult, reset_settings_cache
from furrow.core.orchestrator import MAX_CONSECUTIVE_FAILURES, Orchestrator

GOAL = "ship the feature"


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #
def task(
    task_id: str, *, deps: Sequence[str] = (), status: str = "pending"
) -> TaskModel:
    """A task with a deterministic description, built from plain strings.

    ``status`` is passed as a string on purpose: it exercises the coercion the
    rest of the tests rely on.
    """
    return TaskModel(
        id=task_id,
        description=f"work on {task_id}",
        dependencies=list(deps),
        status=status,
    )


def make_plan(*tasks: TaskModel) -> Plan:
    return Plan(tasks=list(tasks), rationale="because")


def plans_in_order(*plans: Plan) -> Callable[[], Plan]:
    """Hand out ``plans`` in order; the last one answers every later call."""
    remaining = list(plans)

    def factory() -> Plan:
        if not remaining:
            raise AssertionError("the planner was asked for more plans than the test set up")
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return factory


def results_in_order(*results: TestResult) -> Callable[[], TestResult]:
    """Hand out ``results`` in order; the last one answers every later call."""
    remaining = list(results)

    def factory() -> TestResult:
        if not remaining:
            raise AssertionError("the tester was asked for more results than the test set up")
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    return factory


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
def _no_plan() -> Plan:
    raise AssertionError("FakePlanner ran but the test set no plan factory")


def _no_result() -> TestResult:
    raise AssertionError("FakeTester ran but the test set no result factory")


def _summary(item: TaskModel) -> str:
    return f"summary for {item.id}"


def _explode(item: TaskModel) -> str:
    raise RuntimeError(f"worker exploded on {item.id}")


class FakePlanner:
    """Stands in for :class:`~furrow.agents.planner.PlannerAgent`.

    Plans come from :attr:`plan_factory`; every goal string the orchestrator
    passes in is recorded in :attr:`goals`.
    """

    plan_factory: Callable[[], Plan] = _no_plan
    goals: list[str] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def plan(self, goal: str) -> Plan:
        FakePlanner.goals.append(goal)
        return FakePlanner.plan_factory()


class FakeWorker:
    """Stands in for :class:`~furrow.agents.worker.WorkerAgent`.

    :attr:`run_task` decides what a task does: return a summary, or raise.
    """

    run_task: Callable[[TaskModel], str] = _summary
    ran: list[str] = []

    def __init__(self, task: TaskModel, *args: Any, **kwargs: Any) -> None:
        self.task = task

    async def run(self) -> str:
        FakeWorker.ran.append(self.task.id)
        return FakeWorker.run_task(self.task)


class FakeTester:
    """Stands in for :class:`~furrow.agents.tester.TesterAgent`."""

    result_factory: Callable[[], TestResult] = _no_result
    calls: list[tuple[str, list[str]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        FakeTester.calls.append((goal, [item.id for item in tasks]))
        return FakeTester.result_factory()


@pytest.fixture(autouse=True)
def fake_agents(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Swap the real agents for the fakes and start every test with clean state."""
    saved_plan_factory = FakePlanner.plan_factory
    saved_run_task = FakeWorker.run_task
    saved_result_factory = FakeTester.result_factory
    FakePlanner.goals = []
    FakeWorker.ran = []
    FakeTester.calls = []
    monkeypatch.setattr(orchestrator_module, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", FakeTester)
    yield
    FakePlanner.plan_factory = saved_plan_factory
    FakeWorker.run_task = saved_run_task
    FakeTester.result_factory = saved_result_factory


def make_orchestrator(
    goal: str = GOAL,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> Orchestrator:
    """Build an orchestrator with the fake planner already in place.

    No client is passed, so the real ``LLMClient`` is constructed; that only
    reads the settings, it never opens a connection.
    """
    return Orchestrator(goal=goal, on_event=on_event)


# --------------------------------------------------------------------------- #
# _get_tasks / _is_done
# --------------------------------------------------------------------------- #
def test_get_tasks_is_empty_before_a_plan_exists() -> None:
    orch = make_orchestrator()
    assert orch.plan is None
    assert orch._get_tasks() == []


def test_get_tasks_returns_the_planned_tasks() -> None:
    orch = make_orchestrator()
    orch.plan = make_plan(task("a"), task("b"))
    assert [item.id for item in orch._get_tasks()] == ["a", "b"]


def test_get_tasks_tracks_a_replaced_plan() -> None:
    orch = make_orchestrator()
    orch.plan = make_plan(task("a"))
    assert [item.id for item in orch._get_tasks()] == ["a"]
    orch.plan = make_plan(task("b"), task("c"))
    assert [item.id for item in orch._get_tasks()] == ["b", "c"]


def test_is_done_is_false_without_tasks() -> None:
    # The regression: an empty task list made this vacuously True, which is what
    # made run() halt after the very first cycle.
    assert make_orchestrator()._is_done() is False


def test_is_done_is_false_for_a_plan_with_no_tasks() -> None:
    orch = make_orchestrator()
    orch.plan = make_plan()
    assert orch._get_tasks() == []
    assert orch._is_done() is False


def test_is_done_is_false_while_a_task_is_pending() -> None:
    orch = make_orchestrator()
    orch.plan = make_plan(task("a", status="completed"), task("b", status="pending"))
    assert orch._is_done() is False


def test_is_done_is_false_while_a_task_failed() -> None:
    orch = make_orchestrator()
    orch.plan = make_plan(task("a", status="completed"), task("b", status="failed"))
    assert orch._is_done() is False


def test_is_done_is_true_only_when_every_task_completed() -> None:
    orch = make_orchestrator()
    orch.plan = make_plan(task("a", status="completed"), task("b", status="completed"))
    assert orch._is_done() is True


# --------------------------------------------------------------------------- #
# run()
# --------------------------------------------------------------------------- #
async def test_run_executes_a_second_cycle_after_a_failing_one() -> None:
    FakePlanner.plan_factory = plans_in_order(make_plan(task("1")), make_plan(task("2")))

    def fail_task_one(item: TaskModel) -> str:
        if item.id == "1":
            raise RuntimeError("worker exploded on 1")
        return _summary(item)

    FakeWorker.run_task = fail_task_one
    FakeTester.result_factory = results_in_order(
        TestResult(passed=False, summary="one failure", failures=["boom"]),
        TestResult(passed=True, summary="all green", failures=[]),
    )

    orch = make_orchestrator()
    await orch.run()

    # Against the old ``_get_tasks()`` returning ``[]`` this stopped at cycle 1.
    assert orch.cycles == 2
    assert orch.stopped is True
    assert FakeWorker.ran == ["1", "2"]
    assert orch.test_result is not None
    assert orch.test_result.passed is True


async def test_run_honors_the_configured_cycle_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FURROW_MAX_CYCLES", "2")
    # The settings are cached, so the new value is only seen after a reset.
    reset_settings_cache()
    FakePlanner.plan_factory = plans_in_order(make_plan(task("1")))
    FakeWorker.run_task = _explode
    FakeTester.result_factory = results_in_order(
        TestResult(passed=False, summary="still broken", failures=["boom"]),
    )

    orch = make_orchestrator()
    await orch.run()

    assert orch.cycles == 2
    assert orch.stopped is True


async def test_the_original_goal_survives_a_failing_test_cycle() -> None:
    FakePlanner.plan_factory = plans_in_order(make_plan(task("1")), make_plan(task("2")))
    FakeTester.result_factory = results_in_order(
        TestResult(passed=False, summary="s", failures=["boom"]),
        TestResult(passed=True, summary="ok", failures=[]),
    )

    orch = make_orchestrator(goal=GOAL)
    await orch.run()

    assert orch.goal == GOAL
    assert orch.cycles == 2
    # The tester is always handed the pristine goal, repair context included.
    assert FakeTester.calls == [(GOAL, ["1"]), (GOAL, ["2"])]
    # The planner gets the bare goal first, and the goal with an appended repair
    # brief afterwards: the brief is added, never swapped in for the goal.
    assert len(FakePlanner.goals) == 2
    assert FakePlanner.goals[0] == GOAL
    assert FakePlanner.goals[1].startswith(GOAL)
    assert "Fix failing tests:" in FakePlanner.goals[1]
    assert "boom" in FakePlanner.goals[1]


async def test_run_stops_when_the_planner_plans_no_tasks() -> None:
    FakePlanner.plan_factory = plans_in_order(make_plan())

    orch = make_orchestrator()
    await orch.run()

    assert orch.cycles == 1
    assert orch.stopped is True
    assert FakeWorker.ran == []
    assert FakeTester.calls == []


async def test_run_gives_up_after_too_many_consecutive_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FURROW_MAX_CYCLES", "0")  # unlimited
    reset_settings_cache()
    FakePlanner.plan_factory = plans_in_order(make_plan(task("1")))
    FakeWorker.run_task = _explode
    FakeTester.result_factory = results_in_order(
        TestResult(passed=False, summary="still broken", failures=["boom"]),
    )

    orch = make_orchestrator()
    await orch.run()

    # Every failing cycle bumps _consecutive_failures, and _stop_reason halts as
    # soon as it is *strictly* greater than MAX_CONSECUTIVE_FAILURES, so the run
    # ends on cycle MAX_CONSECUTIVE_FAILURES + 1 (4 with the current constant).
    assert orch.cycles == MAX_CONSECUTIVE_FAILURES + 1
    assert orch.stopped is True


def test_max_consecutive_failures_is_three() -> None:
    assert MAX_CONSECUTIVE_FAILURES == 3


# --------------------------------------------------------------------------- #
# _schedule_levels
# --------------------------------------------------------------------------- #
def test_schedule_levels_groups_independent_tasks_together() -> None:
    levels, blocked = make_orchestrator()._schedule_levels([task("a"), task("b"), task("c")])
    assert [sorted(item.id for item in level) for level in levels] == [["a", "b", "c"]]
    assert blocked == []


def test_schedule_levels_runs_a_dependency_first() -> None:
    # "b" is listed first on purpose: the order comes from the dependency graph.
    levels, blocked = make_orchestrator()._schedule_levels(
        [task("b", deps=["a"]), task("a")]
    )
    assert [sorted(item.id for item in level) for level in levels] == [["a"], ["b"]]
    assert blocked == []


def test_schedule_levels_splits_a_dependency_chain() -> None:
    levels, blocked = make_orchestrator()._schedule_levels(
        [task("c", deps=["b"]), task("b", deps=["a"]), task("a")]
    )
    assert [sorted(item.id for item in level) for level in levels] == [["a"], ["b"], ["c"]]
    assert blocked == []


def test_schedule_levels_blocks_an_unknown_dependency() -> None:
    levels, blocked = make_orchestrator()._schedule_levels([task("a", deps=["ghost"])])
    assert levels == []
    assert len(blocked) == 1
    blocked_task, reason = blocked[0]
    assert blocked_task.id == "a"
    assert "unknown dependencies" in reason
    assert "ghost" in reason


def test_schedule_levels_blocks_a_self_dependency() -> None:
    levels, blocked = make_orchestrator()._schedule_levels([task("a", deps=["a"])])
    assert levels == []
    assert len(blocked) == 1
    blocked_task, reason = blocked[0]
    assert blocked_task.id == "a"
    assert "depends on itself" in reason


def test_schedule_levels_blocks_a_dependency_cycle() -> None:
    levels, blocked = make_orchestrator()._schedule_levels(
        [task("a", deps=["b"]), task("b", deps=["a"])]
    )
    assert levels == []
    assert len(blocked) == 2
    assert {item.id for item, _ in blocked} == {"a", "b"}
    assert all("dependency cycle" in reason for _, reason in blocked)


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #
async def test_run_reports_progress_events() -> None:
    events: list[dict[str, Any]] = []
    FakePlanner.plan_factory = plans_in_order(make_plan(task("1")))
    FakeTester.result_factory = results_in_order(
        TestResult(passed=True, summary="ok", failures=[])
    )

    orch = make_orchestrator(on_event=events.append)
    await orch.run()

    types = [event["type"] for event in events]
    assert {"started", "cycle", "plan", "task", "tests", "stopped"} <= set(types)
    # One task, one clean cycle: the order is fixed by run()/_cycle()/_execute().
    assert types == ["started", "cycle", "plan", "task", "tests", "stopped"]
    assert all("type" in event for event in events)
    assert events[0]["goal"] == GOAL
    assert events[-1]["reason"].startswith("Goal complete")


async def test_a_raising_event_callback_cannot_break_the_run() -> None:
    def explode(event: dict[str, Any]) -> None:
        raise RuntimeError("callback exploded")

    FakePlanner.plan_factory = plans_in_order(make_plan(task("1")))
    FakeTester.result_factory = results_in_order(
        TestResult(passed=True, summary="ok", failures=[])
    )

    orch = make_orchestrator(on_event=explode)
    await orch.run()

    assert orch.cycles == 1
    assert orch.stopped is True
    assert orch.test_result is not None
    assert orch.test_result.passed is True
