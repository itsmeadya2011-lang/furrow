"""Behavioural tests for the plan -> execute -> test loop.

Everything here runs in-process: the agents are replaced by scripted stand-ins and
the LLM client refuses to talk to a network, so no test can reach a live model.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
from typing import Any

import pytest
from rich.console import Console

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

GOAL = "build the thing"


@pytest.fixture(autouse=True)
def quiet_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Send the orchestrator's rich output to a throwaway buffer."""
    console = Console(file=io.StringIO())
    monkeypatch.setattr(orchestrator_module, "console", console)
    return console


def make_plan(*task_ids: str) -> Plan:
    return Plan(
        tasks=[TaskModel(id=task_id, description=f"do {task_id}") for task_id in task_ids],
        rationale="scripted plan",
    )


def make_result(passed: bool, summary: str = "scripted", *failures: str) -> TestResult:
    return TestResult(passed=passed, summary=summary, failures=list(failures))


class FakeClient(LLMClient):
    """A real ``LLMClient`` that refuses to do any I/O.

    The orchestrator hands this to its agents; since every agent is stubbed out
    below, ``complete`` should never be reached. If it ever is, failing loudly
    beats silently opening a socket.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings=settings)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        raise AssertionError("orchestrator tests must never call a live LLM")


class ScriptedPlanner:
    """Returns pre-baked plans, one per cycle."""

    def __init__(self, plans: list[Plan], *, repeat_last: bool = False) -> None:
        self._plans = list(plans)
        self._repeat_last = repeat_last
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        # A real yield point: lets the test task regain control mid-cycle, which
        # is what makes cooperative cancellation in the "unlimited" test safe.
        await asyncio.sleep(0)
        if not self._plans:
            return make_plan()  # nothing left to plan -> the run halts
        if len(self._plans) > 1 or not self._repeat_last:
            return self._plans.pop(0)
        return self._plans[0]


class ScriptedTester:
    """Returns pre-baked test verdicts, one per cycle."""

    def __init__(self, results: list[TestResult], *, repeat_last: bool = False) -> None:
        self._results = list(results)
        self._repeat_last = repeat_last
        self.goals: list[str] = []
        self.planned_ids: list[list[str]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.goals.append(goal)
        self.planned_ids.append([task.id for task in tasks])
        if not self._results:
            return make_result(True, "nothing left to test")
        if len(self._results) > 1 or not self._repeat_last:
            return self._results.pop(0)
        return self._results[0]


class ScriptedWorker:
    """Records how many worker coroutines are in flight at the same time."""

    def __init__(self, task: TaskModel, rig: Rig) -> None:
        self.task = task
        self._rig = rig

    async def run(self) -> str:
        rig = self._rig
        rig.in_flight += 1
        rig.max_in_flight = max(rig.max_in_flight, rig.in_flight)
        try:
            if rig.worker_delay:
                await asyncio.sleep(rig.worker_delay)
            return f"done: {self.task.id}"
        finally:
            rig.in_flight -= 1


class EventLog:
    """Callable event hook that remembers everything it was handed."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    @property
    def types(self) -> list[str]:
        return [str(event["type"]) for event in self.events]

    def first(self, event_type: str) -> dict[str, Any]:
        return next(event for event in self.events if event["type"] == event_type)

    def all_of(self, event_type: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == event_type]


class Rig:
    """An ``Orchestrator`` wired up to the scripted agents above."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        plans: list[Plan],
        tests: list[TestResult],
        repeat_plans: bool = False,
        repeat_tests: bool = False,
        worker_delay: float = 0.0,
        max_cycles: int = 0,
        max_consecutive_failures: int = 3,
        max_parallel_tasks: int = 5,
        goal: str = GOAL,
    ) -> None:
        self.planner = ScriptedPlanner(plans, repeat_last=repeat_plans)
        self.tester = ScriptedTester(tests, repeat_last=repeat_tests)
        self.worker_delay = worker_delay
        self.in_flight = 0
        self.max_in_flight = 0
        self.settings = Settings(
            max_parallel_tasks=max_parallel_tasks,
            max_cycles=max_cycles,
            max_consecutive_failures=max_consecutive_failures,
        )
        self.orchestrator = Orchestrator(
            goal=goal, client=FakeClient(self.settings), settings=self.settings
        )
        monkeypatch.setattr(self.orchestrator, "planner", lambda: self.planner)
        monkeypatch.setattr(self.orchestrator, "tester", lambda: self.tester)
        monkeypatch.setattr(self.orchestrator, "worker", self._make_worker)

    def _make_worker(self, task: TaskModel) -> ScriptedWorker:
        return ScriptedWorker(task, self)


# --------------------------------------------------------------------------- #
# 1. _get_tasks() returns the accumulated tasks
# --------------------------------------------------------------------------- #


async def test_get_tasks_returns_accumulated_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression guard for the ``return []`` stub that halted the loop early."""
    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1", "t2"), make_plan("t3")],
        tests=[make_result(True)],
    )

    await rig.orchestrator.run()

    tasks = rig.orchestrator._get_tasks()
    assert [task.id for task in tasks] == ["t1", "t2", "t3"]
    assert tasks is rig.orchestrator.tasks
    # Two productive cycles plus the cycle in which the planner ran dry.
    assert rig.orchestrator.cycles == 3
    assert [task.status for task in tasks] == ["completed", "completed", "completed"]


# --------------------------------------------------------------------------- #
# 2. _merge_tasks replaces rather than duplicates
# --------------------------------------------------------------------------- #


def test_merge_tasks_replaces_task_with_the_same_id() -> None:
    settings = Settings()
    orch = Orchestrator(goal=GOAL, client=FakeClient(settings), settings=settings)
    orch._merge_tasks(
        [TaskModel(id="t1", description="v1"), TaskModel(id="t2", description="v2")]
    )

    orch._merge_tasks(
        [TaskModel(id="t1", description="v1-updated", status="completed", result="ok")]
    )

    assert [task.id for task in orch._get_tasks()] == ["t1", "t2"]
    assert len(orch._get_tasks()) == 2
    assert orch.tasks[0].description == "v1-updated"
    assert orch.tasks[0].status == "completed"
    assert orch.tasks[0].result == "ok"
    assert orch.tasks[1].description == "v2"


# --------------------------------------------------------------------------- #
# 3. max_cycles is respected exactly
# --------------------------------------------------------------------------- #


async def test_max_cycles_runs_exactly_that_many_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    log = EventLog()
    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1")],
        tests=[make_result(True)],
        repeat_plans=True,
        repeat_tests=True,
        max_cycles=2,
    )
    rig.orchestrator.on_event = log

    await rig.orchestrator.run()

    assert rig.orchestrator.cycles == 2
    assert len(rig.planner.goals) == 2
    assert len(rig.tester.goals) == 2
    assert log.types.count("cycle_start") == 2
    assert "cycle budget reached: 2/2 cycles" in log.first("halted")["reason"]


# --------------------------------------------------------------------------- #
# 4. max_consecutive_failures
# --------------------------------------------------------------------------- #


async def test_consecutive_failures_stop_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    log = EventLog()
    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1")],
        tests=[make_result(False, "boom", "test_a failed")],
        repeat_plans=True,
        repeat_tests=True,
        max_cycles=0,
        max_consecutive_failures=2,
    )
    rig.orchestrator.on_event = log

    await rig.orchestrator.run()

    assert rig.orchestrator.cycles == 2
    assert rig.orchestrator.consecutive_failures == 2
    reason = log.first("halted")["reason"]
    assert "failure limit reached: 2 consecutive cycles" in reason
    assert "max_consecutive_failures=2" in reason


async def test_tests_that_eventually_pass_never_trip_the_failure_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = EventLog()
    alternating = [
        make_result(False, "boom", "test_a failed"),
        make_result(True, "green"),
        make_result(False, "boom", "test_a failed"),
        make_result(True, "green"),
        make_result(False, "boom", "test_a failed"),
        make_result(True, "green"),
    ]
    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1")] * 6,
        tests=alternating,
        max_cycles=0,
        max_consecutive_failures=2,
    )
    rig.orchestrator.on_event = log

    await asyncio.wait_for(rig.orchestrator.run(), timeout=10)

    # 6 productive cycles + the cycle in which the planner ran dry.
    assert rig.orchestrator.cycles == 7
    assert rig.orchestrator.consecutive_failures == 0
    assert [event["consecutive_failures"] for event in log.all_of("test_result")] == [
        1,
        0,
        1,
        0,
        1,
        0,
    ]
    assert "goal complete" in log.first("halted")["reason"]


# --------------------------------------------------------------------------- #
# 5. original_goal is immutable, current_goal tracks the failures
# --------------------------------------------------------------------------- #


async def test_original_goal_survives_a_failing_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1")],
        tests=[make_result(False, "boom", "test_a failed", "test_b failed")],
        repeat_plans=True,
        repeat_tests=True,
        max_cycles=0,
        max_consecutive_failures=1,
    )

    await rig.orchestrator.run()

    orch = rig.orchestrator
    assert orch.cycles == 1
    assert orch.original_goal == GOAL
    assert orch.current_goal != GOAL
    assert GOAL in orch.current_goal
    assert "test_a failed" in orch.current_goal
    assert "test_b failed" in orch.current_goal
    # The failing cycle must not have leaked into the plan that produced it.
    assert rig.planner.goals == [GOAL]
    # The tester is always handed the pristine goal, never the fix-it prompt.
    assert rig.tester.goals == [GOAL]


# --------------------------------------------------------------------------- #
# 6. worker parallelism is capped
# --------------------------------------------------------------------------- #


async def test_workers_never_exceed_max_parallel_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    rig = Rig(
        monkeypatch,
        plans=[make_plan(*[f"t{index}" for index in range(6)])],
        tests=[make_result(True)],
        repeat_plans=True,
        repeat_tests=True,
        max_cycles=1,
        max_parallel_tasks=2,
        worker_delay=0.02,
    )

    await rig.orchestrator.run()

    assert rig.orchestrator.cycles == 1
    assert rig.max_in_flight == 2  # the cap is actually reached, so it is a real cap
    assert rig.in_flight == 0  # every worker finished
    assert [task.status for task in rig.orchestrator._get_tasks()] == ["completed"] * 6


# --------------------------------------------------------------------------- #
# 7. / 8. event hooks
# --------------------------------------------------------------------------- #


async def test_on_event_receives_the_full_cycle_lifecycle(monkeypatch: pytest.MonkeyPatch) -> None:
    log = EventLog()
    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1")],
        tests=[make_result(True, "green")],
        repeat_plans=True,
        repeat_tests=True,
        max_cycles=2,
    )
    rig.orchestrator.on_event = log

    await rig.orchestrator.run()

    assert {"cycle_start", "plan", "task_completed", "test_result", "cycle_end"} <= set(log.types)
    assert log.types[0] == "cycle_start"
    assert log.types[-1] == "halted"
    assert log.types.count("cycle_start") == 2
    assert log.types.count("plan") == 2
    assert log.types.count("task_completed") == 2
    assert log.types.count("test_result") == 2
    assert log.types.count("cycle_end") == 2
    assert {event["cycle"] for event in log.events} == {1, 2}

    assert log.first("cycle_start")["goal"] == GOAL
    plan_event = log.first("plan")
    assert plan_event["task_ids"] == ["t1"]
    assert plan_event["count"] == 1
    task_event = log.first("task_completed")
    assert task_event["id"] == "t1"
    assert task_event["status"] == "completed"
    assert task_event["result"].startswith("done: t1")
    test_event = log.first("test_result")
    assert test_event["passed"] is True
    assert test_event["summary"] == "green"
    assert test_event["consecutive_failures"] == 0
    assert log.first("cycle_end")["tasks_completed"] == 1


async def test_a_raising_event_hook_does_not_break_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    def exploding_hook(event: dict[str, Any]) -> None:
        seen.append(str(event["type"]))
        raise RuntimeError("observer blew up")

    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1"), make_plan("t2")],
        tests=[make_result(True)],
        max_cycles=0,
        max_consecutive_failures=3,
    )
    rig.orchestrator.on_event = exploding_hook

    await asyncio.wait_for(rig.orchestrator.run(), timeout=10)

    assert rig.orchestrator.cycles == 3  # 2 productive + the cycle the planner ran dry
    assert [task.id for task in rig.orchestrator._get_tasks()] == ["t1", "t2"]
    assert rig.orchestrator.consecutive_failures == 0
    assert {"cycle_start", "plan", "task_completed", "test_result", "cycle_end"} <= set(seen)
    assert "halted" in seen


# --------------------------------------------------------------------------- #
# 9. max_cycles=0 means unlimited
# --------------------------------------------------------------------------- #


async def test_zero_max_cycles_keeps_looping_until_the_planner_runs_dry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1"), make_plan("t2"), make_plan("t3")],
        tests=[make_result(True)],
        max_cycles=0,
        max_consecutive_failures=1000,
    )

    await asyncio.wait_for(rig.orchestrator.run(), timeout=10)

    assert rig.orchestrator.cycles == 4
    assert [task.id for task in rig.orchestrator._get_tasks()] == ["t1", "t2", "t3"]


async def test_zero_max_cycles_does_not_halt_by_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """The planner never runs dry here, so the run must be stopped by the test."""
    log = EventLog()
    stop = asyncio.Event()
    rig = Rig(
        monkeypatch,
        plans=[make_plan("t1")],
        tests=[make_result(True)],
        repeat_plans=True,
        repeat_tests=True,
        max_cycles=0,
        max_consecutive_failures=1000,
    )

    def hook(event: dict[str, Any]) -> None:
        log(event)
        if event["type"] == "cycle_start" and event["cycle"] >= 4:
            stop.set()

    rig.orchestrator.on_event = hook
    run = asyncio.create_task(rig.orchestrator.run())
    try:
        await asyncio.wait_for(stop.wait(), timeout=5)
    finally:
        run.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run

    assert rig.orchestrator.cycles == 4
    assert len(rig.planner.goals) == 4
    # The 4th cycle was interrupted while planning, so it never got to test.
    assert len(rig.tester.goals) == 3
    assert "halted" not in log.types
