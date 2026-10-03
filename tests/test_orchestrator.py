"""Regression tests for the Orchestrator development loop.

The bug this file exists to prevent: ``_is_done()`` used to call a ``_get_tasks()``
that unconditionally returned ``[]``, so ``completed >= len(tasks)`` evaluated as
``0 >= 0`` -> always ``True``. ``run()`` therefore executed exactly ONE cycle and
printed "Goal complete. Halting.", and the product's whole premise -- a repeated
development loop -- silently did not work. Every test below asserts on the
*number of cycles* the loop actually ran, because a test that only checks the
final task statuses cannot tell one cycle from many.

Every test passes a finite ``max_cycles`` (or relies on a hard-coded halt
condition) so a regression surfaces as a fast assertion failure rather than a
hang.
"""

from __future__ import annotations

import asyncio

import pytest

from furrow.config import Plan, Settings, TaskModel
from furrow.config import TestResult as _TestResult  # aliased: pytest collects Test* names
from furrow.core.orchestrator import (
    MAX_CONSECUTIVE_FAILURES,
    MAX_NO_PROGRESS_CYCLES,
    Orchestrator,
)

# --------------------------------------------------------------------------- #
# Shared fakes / helpers (kept local to this file on purpose -- no conftest).
# --------------------------------------------------------------------------- #


class DummyClient:
    """Stand-in for LLMClient.

    The orchestrator does ``self.client = client or LLMClient(settings=...)``, so
    passing a truthy object here means the real client (and therefore the
    provider SDKs) is never constructed.
    """


def make_settings(max_cycles: int, max_parallel_tasks: int = 5) -> Settings:
    """Settings with an explicit, finite cycle budget and no env-file surprises."""
    return Settings(
        _env_file=None,
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
    )


def plan_of(*specs: tuple[str, str, list[str]]) -> Plan:
    """Build a Plan from ``(id, description, dependencies)`` triples."""
    return Plan(
        tasks=[TaskModel(id=i, description=d, dependencies=list(deps)) for i, d, deps in specs],
        rationale="test plan",
    )


def cycling(specs: tuple[tuple[str, str, list[str]], ...]) -> object:
    """Zero-arg callable returning a *fresh* Plan, so each cycle gets new tasks."""

    def factory() -> Plan:
        return plan_of(*specs)

    return factory


EMPTY_PLAN = Plan(tasks=[], rationale="nothing to do")

PASSING = _TestResult(passed=True, summary="all green")
FAILING = _TestResult(passed=False, summary="1 failed", failures=["boom: test_x failed"])


def install_planner(monkeypatch: pytest.MonkeyPatch, steps: list) -> list[str]:
    """Patch PlannerAgent; return the shared list of goals it was called with.

    ``steps`` is consumed one entry per ``plan()`` call and the last entry is
    reused once exhausted. An entry may be a Plan, a zero-arg callable returning
    a Plan (see :func:`cycling`), or an exception instance to raise.
    """
    goals: list[str] = []

    class FakePlannerAgent:
        def __init__(self, client=None, **kwargs) -> None:
            self.client = client

        async def plan(self, goal: str) -> Plan:
            goals.append(goal)
            step = steps[min(len(goals) - 1, len(steps) - 1)]
            if isinstance(step, BaseException):
                raise step
            return step() if callable(step) else step

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent", FakePlannerAgent)
    return goals


def install_tester(monkeypatch: pytest.MonkeyPatch, results: list[_TestResult]) -> list[tuple]:
    """Patch TesterAgent; return the shared list of ``(goal, task_ids)`` calls."""
    calls: list[tuple] = []

    class FakeTesterAgent:
        def __init__(self, client=None, **kwargs) -> None:
            self.client = client

        async def run(self, goal: str, tasks) -> _TestResult:
            calls.append((goal, [t.id for t in tasks]))
            return results[min(len(calls) - 1, len(results) - 1)]

    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", FakeTesterAgent)
    return calls


def install_worker(monkeypatch: pytest.MonkeyPatch, handler=None) -> list[tuple[str, str]]:
    """Patch WorkerAgent; return the shared ``("start"|"end", task_id)`` log.

    ``handler`` is an optional ``async fn(task, log)`` used to raise, to add
    extra yields, or to measure concurrency.
    """
    log: list[tuple[str, str]] = []

    class FakeWorkerAgent:
        def __init__(self, task=None, client=None, **kwargs) -> None:
            self.task = task
            self.client = client

        async def run(self) -> str:
            log.append(("start", self.task.id))
            if handler is not None:
                await handler(self.task, log)
            else:
                await asyncio.sleep(0)
            log.append(("end", self.task.id))
            return f"wrote {self.task.id}"

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", FakeWorkerAgent)
    return log


def make_orchestrator(goal: str = "ship the feature", **settings_kwargs) -> Orchestrator:
    return Orchestrator(
        goal=goal,
        client=DummyClient(),
        settings=make_settings(**settings_kwargs),
    )


# --------------------------------------------------------------------------- #
# 1. Happy path: one cycle is enough when the work lands and tests are green.
# --------------------------------------------------------------------------- #


async def test_single_cycle_halts_when_all_tasks_complete_and_tests_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goals = install_planner(monkeypatch, [plan_of(("a", "write a", []), ("b", "write b", []))])
    install_tester(monkeypatch, [PASSING])
    install_worker(monkeypatch)

    orch = make_orchestrator(max_cycles=10)
    await orch.run()

    assert orch.cycles == 1, "green work should complete in a single cycle"
    assert len(goals) == 1
    assert [t.status for t in orch._tasks] == ["completed", "completed"]
    assert orch._tests_failed is False


# --------------------------------------------------------------------------- #
# 2. Red tests must schedule another cycle ("will attempt fix in next cycle").
# --------------------------------------------------------------------------- #


async def test_failing_tests_trigger_a_second_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    # Cycle 1 red, cycle 2 green. Fresh plans so the tasks are new objects.
    install_planner(monkeypatch, [cycling((("a", "write a", []),)), cycling((("a", "fix a", []),))])
    tester_calls = install_tester(monkeypatch, [FAILING, PASSING])
    install_worker(monkeypatch)

    orch = make_orchestrator(max_cycles=10)
    await orch.run()

    assert orch.cycles == 2, "a red suite must buy exactly one more cycle, then stop on green"
    assert len(tester_calls) == 2
    assert orch._tests_failed is False


# --------------------------------------------------------------------------- #
# 3. A raising task is captured as a failed task, not an unhandled crash.
# --------------------------------------------------------------------------- #


async def test_raising_task_is_marked_failed_and_ends_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(task, log) -> None:
        raise RuntimeError("worker exploded")

    task = TaskModel(id="a", description="explode", status="pending")
    install_planner(monkeypatch, [Plan(tasks=[task], rationale="one task")])
    install_tester(monkeypatch, [PASSING])
    install_worker(monkeypatch, handler=boom)

    orch = make_orchestrator(max_cycles=10)
    await orch.run()

    assert task.status == "failed"
    assert task.result is not None and "worker exploded" in task.result
    assert orch.cycles == 1, "a failed task is terminal; the loop must stop, not retry forever"
    assert orch._is_done() is True


# --------------------------------------------------------------------------- #
# 4. THE REGRESSION TEST: an empty plan is not success.
#    With the old ``_get_tasks() -> []`` bug this run halted at cycle 1.
# --------------------------------------------------------------------------- #


async def test_empty_plan_is_not_treated_as_done(monkeypatch: pytest.MonkeyPatch) -> None:
    goals = install_planner(monkeypatch, [EMPTY_PLAN])
    install_tester(monkeypatch, [PASSING])
    install_worker(monkeypatch)

    orch = make_orchestrator(max_cycles=3)
    await orch.run()

    assert orch.cycles == 3, "an empty plan must not report completion at cycle 1"
    assert len(goals) == 3, "the loop must keep re-planning while plans come back empty"
    assert orch._tasks == []


# --------------------------------------------------------------------------- #
# 5. max_cycles is an exact budget (0 means unlimited, so never test 0 here).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("budget", [1, 2, 4])
async def test_max_cycles_is_honoured_exactly(monkeypatch: pytest.MonkeyPatch, budget: int) -> None:
    # Tasks always complete and the suite is always red, so neither the
    # no-progress counter (progressed=True resets it) nor _is_done can stop the
    # loop -- only the budget can. budget > MAX_NO_PROGRESS_CYCLES on purpose.
    install_planner(monkeypatch, [cycling((("a", "keep going", []),))])
    install_tester(monkeypatch, [FAILING])
    install_worker(monkeypatch)

    orch = make_orchestrator(max_cycles=budget)
    await orch.run()

    assert orch.cycles == budget


# --------------------------------------------------------------------------- #
# 6. A permanently broken planner cannot spin the loop forever.
# --------------------------------------------------------------------------- #


async def test_consecutive_cycle_failures_halt_the_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    install_planner(monkeypatch, [RuntimeError("provider is down")])
    install_tester(monkeypatch, [PASSING])
    install_worker(monkeypatch)

    orch = make_orchestrator(max_cycles=100)
    await orch.run()

    assert MAX_CONSECUTIVE_FAILURES == 3
    assert orch.cycles == MAX_CONSECUTIVE_FAILURES, (
        "the loop must halt on the 3rd consecutive failure, long before max_cycles"
    )


# --------------------------------------------------------------------------- #
# 7. No-progress (all-empty plans) halts before the cycle budget.
# --------------------------------------------------------------------------- #


async def test_no_progress_cycles_halt_before_the_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    install_planner(monkeypatch, [EMPTY_PLAN])
    install_tester(monkeypatch, [PASSING])
    install_worker(monkeypatch)

    orch = make_orchestrator(max_cycles=50)
    await orch.run()

    assert MAX_NO_PROGRESS_CYCLES == 3
    assert orch.cycles == MAX_NO_PROGRESS_CYCLES, (
        "3 fruitless cycles must halt the run instead of burning the whole budget"
    )


# --------------------------------------------------------------------------- #
# 8. The goal is immutable; test failures become the next plan's feedback.
# --------------------------------------------------------------------------- #


async def test_failure_feedback_feeds_next_plan_and_goal_is_never_mutated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goal = "make the parser faster"
    goals = install_planner(monkeypatch, [cycling((("a", "do it", []),))])
    install_tester(monkeypatch, [FAILING, PASSING])
    install_worker(monkeypatch)

    orch = make_orchestrator(goal=goal, max_cycles=10)
    await orch.run()

    assert orch.goal == goal, "the orchestrator must never rewrite its own goal"
    assert len(goals) == 2, "red cycle 1 must re-plan for cycle 2"

    # Cycle 1 has no feedback yet, so the planner sees the bare goal.
    assert goals[0] == goal
    # Cycle 2 is handed the failure text ahead of the unchanged goal.
    assert "boom: test_x failed" in goals[1]
    assert goal in goals[1]
    assert goals[1].startswith("Fix failing tests:")
    assert goals[1] != goal


# --------------------------------------------------------------------------- #
# 9. Dependency waves.
# --------------------------------------------------------------------------- #


async def test_dependent_task_starts_only_after_its_dependency_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_b = TaskModel(id="b", description="use a", dependencies=["a"])
    plan = Plan(
        tasks=[TaskModel(id="a", description="base", dependencies=[]), task_b],
        rationale="waves",
    )
    install_planner(monkeypatch, [plan])
    install_tester(monkeypatch, [PASSING])
    log = install_worker(monkeypatch)

    orch = make_orchestrator(max_cycles=5)
    await orch.run()

    starts = [tid for kind, tid in log if kind == "start"]
    ends = [tid for kind, tid in log if kind == "end"]

    # Wave membership: "a" runs alone in wave 1, "b" alone in wave 2.
    assert starts == ["a", "b"]
    assert ends == ["a", "b"]
    assert log.index(("start", "b")) > log.index(("end", "a")), "b must not start before a finished"
    assert task_b.status == "completed"


async def test_task_depending_on_failed_task_is_skipped_and_never_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def boom(task, log) -> None:
        raise RuntimeError("upstream broke")

    task_d = TaskModel(id="d", description="upstream", dependencies=[])
    task_c = TaskModel(id="c", description="needs d", dependencies=["d"])
    plan = Plan(tasks=[task_d, task_c], rationale="chain")

    install_planner(monkeypatch, [plan])
    install_tester(monkeypatch, [PASSING])
    log = install_worker(monkeypatch, handler=boom)

    orch = make_orchestrator(max_cycles=5)
    await orch.run()

    assert task_d.status == "failed"
    assert task_c.status == "skipped"
    assert task_c.result == "skipped: dependency did not complete"
    assert [tid for _, tid in log] == ["d"], "c's worker must never be invoked"
    assert orch.cycles == 1, "a failed task is terminal"


# --------------------------------------------------------------------------- #
# 10. max_parallel_tasks is a hard ceiling on concurrent workers.
# --------------------------------------------------------------------------- #


async def test_max_parallel_tasks_caps_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    in_flight = 0
    high_water = 0

    async def track(task, log) -> None:
        nonlocal in_flight, high_water
        in_flight += 1
        high_water = max(high_water, in_flight)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        in_flight -= 1

    specs = tuple((f"t{i}", f"task {i}", []) for i in range(6))
    install_planner(monkeypatch, [cycling(specs)])
    install_tester(monkeypatch, [PASSING])
    log = install_worker(monkeypatch, handler=track)

    orch = make_orchestrator(max_cycles=3, max_parallel_tasks=2)
    await orch.run()

    assert len(log) == 12, "all six tasks must run and finish"
    assert high_water <= 2, f"semaphore let {high_water} workers run at once (limit 2)"
    assert high_water == 2, "with 6 runnable tasks the limit should actually be reached"
    assert in_flight == 0
    assert all(t.status == "completed" for t in orch._tasks)


# --------------------------------------------------------------------------- #
# 11. A productive cycle resets the no-progress counter.
# --------------------------------------------------------------------------- #


async def test_completing_a_task_resets_the_no_progress_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Alternating: cycle 1 empty, cycle 2 productive-but-still-red, cycle 3 empty,
    # ... Each empty cycle only pushes `stalled` to 1 because cycle 2 resets it.
    # Without the reset the run would halt on cycle 3.
    n_plans = 0

    def alternating() -> Plan:
        nonlocal n_plans
        n_plans += 1
        return EMPTY_PLAN if n_plans % 2 else plan_of(("a", "work", []))

    goals = install_planner(monkeypatch, [alternating])
    install_tester(monkeypatch, [FAILING])
    install_worker(monkeypatch)

    orch = make_orchestrator(max_cycles=6)
    await orch.run()

    assert len(goals) == 6, "the loop should reach the budget, past the 3-cycle stall limit"
    assert orch.cycles == 6
    assert orch.cycles > MAX_NO_PROGRESS_CYCLES
