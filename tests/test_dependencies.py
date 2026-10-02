"""Tests for dependency-aware scheduling in :mod:`furrow.core.orchestrator`.

``TaskModel.dependencies`` is part of the schema and ``PLANNER_PROMPT`` asks the
model for it, so the orchestrator has to honour it: ``WorkerAgent`` writes whole
files through ``client.write_file``, and two workers editing one file at the same
time silently discard one of the two edits.

The small rig below deliberately mirrors ``tests/test_orchestrator.py`` (same
fake client, same scripted agents, same quiet-console fixture) so the two files
read as one. What is new here is the *ordered* start/end log: ordering claims are
only checkable if the log records when each worker actually entered and left
``run()``, rather than just counting concurrency the way the other file does.

Every test runs one cycle with ``max_cycles=1`` so the interesting part is the
single ``_execute`` pass, and every cycle-shaped case is wrapped in
``asyncio.wait_for`` so a scheduler that forgets to terminate fails the test
instead of hanging the suite.
"""

from __future__ import annotations

import asyncio
import io
from typing import Any

import pytest
from rich.console import Console

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import SKIPPED, Orchestrator
from furrow.llm import LLMClient

GOAL = "build the thing"

#: Generous next to the ~0.02s these workers sleep, and still far below the
#: point where a wedged scheduler is a hung suite.
DEADLINE = 10


@pytest.fixture(autouse=True)
def quiet_console(monkeypatch: pytest.MonkeyPatch) -> Console:
    """Send the orchestrator's rich output to a throwaway buffer."""
    console = Console(file=io.StringIO())
    monkeypatch.setattr(orchestrator_module, "console", console)
    return console


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #


def task(task_id: str, *dependencies: str) -> TaskModel:
    """One planned task, with the ids it must wait for."""
    return TaskModel(
        id=task_id, description=f"do {task_id}", dependencies=list(dependencies)
    )


def make_plan(*tasks: TaskModel) -> Plan:
    return Plan(tasks=list(tasks), rationale="scripted plan")


def make_result(passed: bool = True, summary: str = "scripted") -> TestResult:
    return TestResult(passed=passed, summary=summary, failures=[])


class FakeClient(LLMClient):
    """A real ``LLMClient`` that refuses to do any I/O.

    Every agent is stubbed below, so ``complete`` should never be reached. If it
    ever is, failing loudly beats silently opening a socket.
    """

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings=settings)

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        raise AssertionError("dependency tests must never call a live LLM")


class ScriptedPlanner:
    """Returns pre-baked plans, one per cycle."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self._used = False
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        # A real yield point, same as the sibling test file: it gives a running
        # test task control back mid-cycle.
        await asyncio.sleep(0)
        if self._used:
            return make_plan()  # nothing left to plan -> the run halts
        self._used = True
        return self._plan


class ScriptedTester:
    """Always passes; these tests are about scheduling, not verdicts."""

    def __init__(self) -> None:
        self.goals: list[str] = []
        self.planned_ids: list[list[str]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.goals.append(goal)
        self.planned_ids.append([t.id for t in tasks])
        return make_result()


class RecordingWorker:
    """Logs when it enters and leaves ``run()``, and tracks concurrency."""

    def __init__(self, task: TaskModel, rig: Rig) -> None:
        self.task = task
        self._rig = rig

    async def run(self) -> str:
        rig = self._rig
        rig.log.append(("start", self.task.id))
        rig.in_flight += 1
        rig.max_in_flight = max(rig.max_in_flight, rig.in_flight)
        try:
            if rig.worker_delay:
                await asyncio.sleep(rig.worker_delay)
            if self.task.id in rig.failing:
                raise RuntimeError(f"worker for {self.task.id} exploded")
            return f"done: {self.task.id}"
        finally:
            rig.in_flight -= 1
            # Logged even on failure, so the ordering log stays total.
            rig.log.append(("end", self.task.id))


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
    """An ``Orchestrator`` wired to the scripted agents above, single cycle."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tasks: list[TaskModel],
        *,
        max_parallel_tasks: int = 5,
        worker_delay: float = 0.0,
        failing: set[str] | None = None,
    ) -> None:
        self.planner = ScriptedPlanner(make_plan(*tasks))
        self.tester = ScriptedTester()
        self.log: list[tuple[str, str]] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self.worker_delay = worker_delay
        self.failing: set[str] = set(failing or ())
        self.settings = Settings(
            max_parallel_tasks=max_parallel_tasks,
            # One cycle: these tests are about a single _execute pass.
            max_cycles=1,
            max_consecutive_failures=3,
        )
        self.orchestrator = Orchestrator(
            goal=GOAL, client=FakeClient(self.settings), settings=self.settings
        )
        self.events = EventLog()
        self.orchestrator.on_event = self.events
        monkeypatch.setattr(self.orchestrator, "planner", lambda: self.planner)
        monkeypatch.setattr(self.orchestrator, "tester", lambda: self.tester)
        monkeypatch.setattr(self.orchestrator, "worker", self._make_worker)

    def _make_worker(self, task: TaskModel) -> RecordingWorker:
        return RecordingWorker(task, self)

    async def run(self) -> None:
        await asyncio.wait_for(self.orchestrator.run(), timeout=DEADLINE)

    @property
    def statuses(self) -> dict[str, str]:
        return {t.id: t.status for t in self.orchestrator._get_tasks()}

    @property
    def results(self) -> dict[str, str]:
        return {t.id: (t.result or "") for t in self.orchestrator._get_tasks()}

    @property
    def started(self) -> list[str]:
        return [task_id for kind, task_id in self.log if kind == "start"]

    @property
    def ended(self) -> list[str]:
        return [task_id for kind, task_id in self.log if kind == "end"]

    def at(self, entry: tuple[str, str]) -> int:
        return self.log.index(entry)

    def assert_before(self, first: tuple[str, str], second: tuple[str, str]) -> None:
        assert self.at(first) < self.at(second), (
            f"{first} should precede {second}, log was {self.log}"
        )


# --------------------------------------------------------------------------- #
# 1. no dependencies -> parallelism is not the price of ordering
# --------------------------------------------------------------------------- #


async def test_plan_without_dependencies_keeps_full_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard against "fixing" ordering by serialising everything.

    A plan with no dependencies must still saturate the cap: otherwise the
    dependency fix would have thrown away the reason ``max_parallel_tasks``
    exists, and a 40-task plan would run one task at a time.
    """
    rig = Rig(
        monkeypatch,
        [task(f"t{index}") for index in range(6)],
        max_parallel_tasks=2,
        worker_delay=0.02,
    )

    await rig.run()

    assert rig.orchestrator.cycles == 1
    assert rig.max_in_flight == 2  # the cap is reached, so it is a real cap
    assert rig.in_flight == 0
    assert rig.started == [f"t{index}" for index in range(6)]
    assert rig.statuses == {f"t{index}": "completed" for index in range(6)}
    assert rig.events.all_of("dependency_warning") == []


# --------------------------------------------------------------------------- #
# 2. a dependent never starts before its dependency is done
# --------------------------------------------------------------------------- #


async def test_a_dependent_never_starts_before_its_dependency_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(
        monkeypatch,
        [task("base"), task("middle", "base"), task("top", "middle")],
        max_parallel_tasks=4,
        worker_delay=0.02,
    )

    await rig.run()

    # The whole chain, in the order the workers actually ran.
    assert rig.log == [
        ("start", "base"),
        ("end", "base"),
        ("start", "middle"),
        ("end", "middle"),
        ("start", "top"),
        ("end", "top"),
    ]
    assert rig.statuses == {"base": "completed", "middle": "completed", "top": "completed"}
    assert rig.results["top"].startswith("done: top")


# --------------------------------------------------------------------------- #
# 3. a failed dependency skips (and reports) its dependents
# --------------------------------------------------------------------------- #


async def test_a_failed_dependency_skips_and_reports_the_dependent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(
        monkeypatch,
        [task("bad"), task("child", "bad")],
        max_parallel_tasks=4,
        worker_delay=0.01,
        failing={"bad"},
    )

    await rig.run()

    assert rig.statuses == {"bad": "failed", "child": SKIPPED}
    # The dependent never ran at all.
    assert rig.started == ["bad"]
    # ...and it is reported rather than silently dropped.
    reported = {event["id"]: event for event in rig.events.all_of("task_completed")}
    assert set(reported) == {"bad", "child"}
    assert reported["child"]["status"] == SKIPPED
    assert "unmet dependency" in reported["child"]["result"]
    assert "bad" in reported["child"]["result"]
    assert "unmet dependency" in rig.results["child"]


async def test_skips_cascade_to_grandchildren_of_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task blocked by a *skipped* task is itself skipped, never run."""
    rig = Rig(
        monkeypatch,
        [task("bad"), task("child", "bad"), task("grandchild", "child")],
        max_parallel_tasks=4,
        failing={"bad"},
    )

    await rig.run()

    assert rig.statuses == {
        "bad": "failed",
        "child": SKIPPED,
        "grandchild": SKIPPED,
    }
    assert rig.started == ["bad"]
    reported = {event["id"] for event in rig.events.all_of("task_completed")}
    assert reported == {"bad", "child", "grandchild"}


# --------------------------------------------------------------------------- #
# 4. / 5. malformed graphs terminate instead of wedging the scheduler
# --------------------------------------------------------------------------- #


async def test_a_dependency_cycle_terminates_and_reports_the_blocked_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``a -> b -> a``: no task is ever runnable, so both must be reported.

    Bounded by ``Rig.run``'s ``wait_for``: a ready-queue that simply waits for a
    dependency to complete would hang here forever, and a hanging test is worse
    than a failing one.
    """
    rig = Rig(monkeypatch, [task("a", "b"), task("b", "a")], max_parallel_tasks=4)

    await rig.run()  # the deadline is the assertion

    assert rig.started == []
    assert rig.statuses == {"a": SKIPPED, "b": SKIPPED}
    warning = rig.events.first("dependency_warning")
    assert warning["reason"] == "cycle"
    assert set(warning["tasks"]) == {"a", "b"}
    for task_id in ("a", "b"):
        assert "cycle" in rig.results[task_id]
        assert task_id in rig.results[task_id]  # names what is blocking it


async def test_a_self_dependency_terminates_and_is_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(monkeypatch, [task("a", "a")], max_parallel_tasks=4)

    await rig.run()  # the deadline is the assertion

    assert rig.started == []
    assert rig.statuses == {"a": SKIPPED}
    assert "cycle" in rig.results["a"]
    assert rig.events.first("dependency_warning")["reason"] == "cycle"
    assert rig.events.first("task_completed")["status"] == SKIPPED


async def test_a_cycle_does_not_take_unrelated_tasks_down_with_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cyclic pair is skipped; the sound task beside it still runs."""
    rig = Rig(
        monkeypatch,
        [task("a", "b"), task("b", "a"), task("fine")],
        max_parallel_tasks=4,
        worker_delay=0.01,
    )

    await rig.run()

    assert rig.started == ["fine"]
    assert rig.statuses == {"a": SKIPPED, "b": SKIPPED, "fine": "completed"}


# --------------------------------------------------------------------------- #
# 6. a dependency on a task that is not in the plan
# --------------------------------------------------------------------------- #


async def test_an_unknown_dependency_is_recorded_and_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(monkeypatch, [task("a", "999")], max_parallel_tasks=4)

    await rig.run()  # the deadline is the assertion

    # "999" is not in this plan, so it can never be satisfied: ignoring the
    # reference is the only non-wedging option.
    assert rig.started == ["a"]
    assert rig.statuses == {"a": "completed"}
    assert rig.results["a"].startswith("done: a")
    warning = rig.events.first("dependency_warning")
    assert warning["reason"] == "unknown_dependency"
    assert warning["refs"] == {"a": ["999"]}
    # The run itself completed normally, no halt, no error.
    assert rig.events.first("test_result")["passed"] is True
    assert rig.orchestrator.consecutive_failures == 0


# --------------------------------------------------------------------------- #
# 7. diamond
# --------------------------------------------------------------------------- #


async def test_a_diamond_waits_for_both_arms_before_the_join(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # c -> a, c -> b, (a + b) -> d
    rig = Rig(
        monkeypatch,
        [task("c"), task("a", "c"), task("b", "c"), task("d", "a", "b")],
        max_parallel_tasks=2,
        worker_delay=0.02,
    )

    await rig.run()

    # c first and alone, then both arms at once, then the join.
    assert rig.log[0] == ("start", "c")
    assert rig.log[1] == ("end", "c")
    assert [task_id for _, task_id in rig.log[2:4]] == ["a", "b"]
    assert rig.max_in_flight == 2  # the arms really did run in parallel
    rig.assert_before(("end", "a"), ("start", "d"))
    rig.assert_before(("end", "b"), ("start", "d"))
    assert rig.statuses == {
        "c": "completed",
        "a": "completed",
        "b": "completed",
        "d": "completed",
    }


# --------------------------------------------------------------------------- #
# 8. the cap still holds when the levels are uneven
# --------------------------------------------------------------------------- #


async def test_the_parallel_cap_holds_across_uneven_dependency_levels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Four roots, two mid-level tasks, one join: only 2 may ever be in flight.
    rig = Rig(
        monkeypatch,
        [
            task("a"),
            task("b"),
            task("c"),
            task("d"),
            task("e", "a", "b"),
            task("f", "c", "d"),
            task("g", "e", "f"),
        ],
        max_parallel_tasks=2,
        worker_delay=0.01,
    )

    await rig.run()

    assert rig.max_in_flight == 2
    assert rig.in_flight == 0
    assert sorted(rig.started) == ["a", "b", "c", "d", "e", "f", "g"]
    for upstream, downstream in (
        ("a", "e"),
        ("b", "e"),
        ("c", "f"),
        ("d", "f"),
        ("e", "g"),
        ("f", "g"),
    ):
        rig.assert_before(("end", upstream), ("start", downstream))
    assert all(status == "completed" for status in rig.statuses.values())


# --------------------------------------------------------------------------- #
# 9. mixed plan
# --------------------------------------------------------------------------- #


async def test_independent_tasks_start_immediately_alongside_dependent_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rig = Rig(
        monkeypatch,
        [
            task("free_a"),
            task("child", "base"),
            task("free_b"),
            task("base"),
            task("tail", "child"),
        ],
        max_parallel_tasks=3,
        worker_delay=0.01,
    )

    await rig.run()

    # The two tasks with no dependencies go first; "child" cannot start until
    # "base" runs, and it does not hold a slot open while waiting.
    assert rig.started[:2] == ["free_a", "free_b"]
    assert rig.log.index(("start", "child")) > rig.log.index(("start", "free_a"))
    rig.assert_before(("end", "base"), ("start", "child"))
    rig.assert_before(("end", "child"), ("start", "tail"))
    assert rig.statuses == {
        "free_a": "completed",
        "child": "completed",
        "free_b": "completed",
        "base": "completed",
        "tail": "completed",
    }
