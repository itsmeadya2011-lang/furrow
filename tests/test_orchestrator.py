"""Offline tests for :mod:`furrow.core.orchestrator`.

Every agent the orchestrator drives -- planner, worker and tester -- is replaced by a
stub class driven by a per-test script, and the LLM client is a fake object holding
nothing but a ``settings`` attribute.  No test in this module opens a socket, reads an
API key, shells out to a test runner or touches a real repository: the workspace is a
pytest ``tmp_path``.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import pytest
from rich.console import Console

from furrow.agents.planner import PlannerError
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.events import Event, EventKind
from furrow.core.orchestrator import Orchestrator

GOAL = "make the widget factory great"


# ---------------------------------------------------------------------------
# helpers: plans, test results and the mutable script the stub agents read
# ---------------------------------------------------------------------------


def make_task(
    task_id: str,
    description: str | None = None,
    *,
    dependencies: tuple[str, ...] = (),
    files: tuple[str, ...] = (),
) -> TaskModel:
    """Build one planned task with predictable defaults."""
    return TaskModel(
        id=task_id,
        description=description if description is not None else f"work on {task_id}",
        dependencies=list(dependencies),
        files=list(files),
    )


def make_plan(*tasks: TaskModel, rationale: str = "stub plan") -> Plan:
    """Build a plan out of ``tasks``."""
    return Plan(tasks=list(tasks), rationale=rationale)


def passing(summary: str = "5 passed") -> TestResult:
    """Build a green :class:`~furrow.config.TestResult`."""
    return TestResult(passed=True, summary=summary, failures=[], exit_code=0, command="pytest -q")


def failing(*failures: str, summary: str = "1 failed") -> TestResult:
    """Build a red :class:`~furrow.config.TestResult`."""
    return TestResult(
        passed=False,
        summary=summary,
        failures=list(failures) or ["tests/test_widget.py::test_size failed"],
        exit_code=1,
        command="pytest -q",
    )


@dataclass
class StubAgents:
    """The script the stub agents consult, plus a log of every call they received.

    Attributes:
        planner_script: Items handed to the planner in order; the last one repeats.
            An item is a :class:`Plan` to return, an exception to raise, or a callable
            ``(goal, *, focus, context) -> Plan``.
        worker_hook: Called with the :class:`TaskModel`; its return value (awaited when
            awaitable) is the worker's summary.  Raising marks the task failed.
        tester_script: Items handed to the tester in order; the last one repeats, with
            the same item kinds as ``planner_script`` minus the callable form.
        planner_calls: ``{"goal", "focus", "context"}`` per planning call.
        worker_calls: The id of every task a worker was asked to run, in order.
        tester_calls: ``{"goal", "task_ids"}`` per testing call.
    """

    planner_script: list[Any] = field(default_factory=list)
    worker_hook: Callable[[TaskModel], Any] = lambda task: f"implemented {task.description}"
    tester_script: list[Any] = field(default_factory=list)

    planner_calls: list[dict[str, Any]] = field(default_factory=list)
    worker_calls: list[str] = field(default_factory=list)
    tester_calls: list[dict[str, Any]] = field(default_factory=list)


def _next_script_item(script: list[Any], index: int, agent: str) -> Any:
    """Return script item ``index``, repeating the last item once the script runs out."""
    if not script:
        raise AssertionError(f"the stub {agent} was called but its script is empty")
    return script[index] if index < len(script) else script[-1]


class _StubPlanner:
    """Stand-in for :class:`furrow.agents.planner.PlannerAgent`."""

    def __init__(self, *, client: Any, settings: Any, state: StubAgents) -> None:
        """Remember the agents state so every call reads the live script."""
        self.client = client
        self.settings = settings
        self._state = state

    async def plan(self, goal: str, *, focus: str | None = None, context: str | None = None) -> Plan:
        """Return (or raise) the next scripted plan."""
        state = self._state
        state.planner_calls.append({"goal": goal, "focus": focus, "context": context})
        item = _next_script_item(state.planner_script, len(state.planner_calls) - 1, "planner")
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(goal, focus=focus, context=context)
        return item


class _StubWorker:
    """Stand-in for :class:`furrow.agents.worker.WorkerAgent`."""

    def __init__(self, *, task: TaskModel, client: Any, settings: Any, state: StubAgents) -> None:
        """Remember the task and the agents state."""
        self.task = task
        self.client = client
        self.settings = settings
        self._state = state

    async def run(self, *, on_event: Any = None) -> str:
        """Delegate to the worker hook, awaiting it when it is a coroutine function."""
        self._state.worker_calls.append(self.task.id)
        outcome = self._state.worker_hook(self.task)
        if asyncio.iscoroutine(outcome) or isinstance(outcome, asyncio.Future):
            return await outcome
        return outcome


class _StubTester:
    """Stand-in for :class:`furrow.agents.tester.TesterAgent`."""

    def __init__(self, *, client: Any, settings: Any, state: StubAgents) -> None:
        """Remember the agents state so every call reads the live script."""
        self.client = client
        self.settings = settings
        self._state = state

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        """Return (or raise) the next scripted test result."""
        state = self._state
        state.tester_calls.append({"goal": goal, "task_ids": [task.id for task in tasks]})
        item = _next_script_item(state.tester_script, len(state.tester_calls) - 1, "tester")
        if isinstance(item, BaseException):
            raise item
        return item


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


class FakeClient:
    """The smallest object the orchestrator accepts as an LLM client."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Store the settings; no provider client is ever built."""
        self.settings = settings


class EventCollector:
    """A synchronous ``on_event`` callback that keeps every event it is handed."""

    def __init__(self) -> None:
        """Start with an empty event list."""
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        """Record one event (sync on purpose: ``emit`` must support sync callbacks)."""
        self.events.append(event)

    def kinds(self) -> list[EventKind]:
        """Return the recorded kinds, in order."""
        return [event.kind for event in self.events]

    def of(self, kind: EventKind) -> list[Event]:
        """Return the recorded events of one kind, in order."""
        return [event for event in self.events if event.kind is kind]

    def messages_of(self, kind: EventKind) -> list[str]:
        """Return the messages of the recorded events of one kind."""
        return [event.message for event in self.of(kind)]


@pytest.fixture
def settings(tmp_path: Any) -> Settings:
    """Return settings rooted at a throwaway workspace and otherwise predictable."""
    return Settings(
        workspace=tmp_path,
        max_cycles=0,
        max_parallel_tasks=5,
        planner_max_tasks=5,
    )


@pytest.fixture
def fake_client(settings: Settings) -> FakeClient:
    """Return a fake LLM client so no provider is ever contacted."""
    return FakeClient(settings)


@pytest.fixture
def events() -> EventCollector:
    """Return an event collector usable as ``on_event``."""
    return EventCollector()


@pytest.fixture
def stub_agents(monkeypatch: pytest.MonkeyPatch) -> StubAgents:
    """Replace the three agent classes the orchestrator looks up in its own module."""
    state = StubAgents()

    def planner(**kwargs: Any) -> _StubPlanner:
        return _StubPlanner(state=state, **kwargs)

    def worker(**kwargs: Any) -> _StubWorker:
        return _StubWorker(state=state, **kwargs)

    def tester(**kwargs: Any) -> _StubTester:
        return _StubTester(state=state, **kwargs)

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", planner)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", worker)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", tester)
    return state


def build(
    goal: str = GOAL,
    *,
    client: Any = None,
    settings: Any = None,
    on_event: Any = None,
) -> Orchestrator:
    """Build an orchestrator with the documented defaults for the tests."""
    return Orchestrator(goal, client=client, settings=settings, on_event=on_event)


# ---------------------------------------------------------------------------
# the regression this codebase shipped with: completing means work happened
# ---------------------------------------------------------------------------


async def test_two_successful_tasks_and_green_tests_complete_the_goal_in_one_cycle(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A complete run must have executed the planned work, not just declared it done."""
    stub_agents.planner_script = [make_plan(make_task("t1"), make_task("t2"))]
    stub_agents.worker_hook = lambda task: f"wrote {task.id}"
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await orchestrator.run()

    assert result.completed is True
    assert result.reason == "goal complete"
    assert result.cycles == 1
    assert result.tasks_completed == 2
    assert sorted(stub_agents.worker_calls) == ["t1", "t2"]
    assert result.tests_passed is True


async def test_completion_is_not_reported_before_the_workers_ran(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """The goal must not be reached without the planned work having executed."""
    stub_agents.planner_script = [make_plan(make_task("t1", "write the widget"))]
    stub_agents.tester_script = [passing()]
    worker_calls: list[str] = []

    def hook(task: TaskModel) -> str:
        worker_calls.append(task.id)
        return "done"

    stub_agents.worker_hook = hook
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert result.tasks_completed == 1
    assert worker_calls == ["t1"]
    assert [task.status for task in orchestrator.plan.tasks] == [TaskStatus.COMPLETED]


# ---------------------------------------------------------------------------
# the goal is never rewritten; failures live in ``focus``
# ---------------------------------------------------------------------------


async def test_failing_tests_drive_a_second_cycle_and_leave_the_goal_untouched(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """A red suite plans again for the same goal instead of rewriting the goal itself."""
    stub_agents.planner_script = [
        make_plan(make_task("t1", "write the parser")),
        make_plan(make_task("t2", "fix the parser")),
    ]
    stub_agents.tester_script = [failing("tests/test_parser.py::test_parse failed"), passing()]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert result.cycles == 2
    assert result.completed is True
    assert orchestrator.goal == GOAL
    assert "Fix failing tests" not in orchestrator.goal
    assert orchestrator.focus == "tests/test_parser.py::test_parse failed"


async def test_the_second_cycle_hands_the_planner_the_original_goal_and_the_failure(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """The planner is given the original goal plus the failure text as focus."""
    stub_agents.planner_script = [make_plan(make_task("t1", "first")), make_plan(make_task("t2"))]
    stub_agents.tester_script = [failing("boom: expected 3 got 4"), passing()]
    orchestrator = build(client=fake_client, settings=settings)

    await orchestrator.run()

    assert len(stub_agents.planner_calls) == 2
    assert stub_agents.planner_calls[1]["goal"] == GOAL
    assert "boom: expected 3 got 4" in stub_agents.planner_calls[1]["focus"]
    assert "boom: expected 3 got 4" in stub_agents.planner_calls[1]["context"]


# ---------------------------------------------------------------------------
# cycle limits
# ---------------------------------------------------------------------------


async def test_max_cycles_two_stops_after_exactly_two_cycles(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """``max_cycles=2`` runs two cycles and stops on the limit, not before."""
    settings.max_cycles = 2
    stub_agents.planner_script = [make_plan(make_task("t1", "one"))]
    stub_agents.tester_script = [failing("still red")]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert result.cycles == 2
    assert result.reason == "cycle limit reached"
    assert result.completed is False
    assert len(stub_agents.planner_calls) == 2
    assert len(stub_agents.tester_calls) == 2


async def test_max_cycles_zero_is_not_a_cap(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """``max_cycles=0`` means unlimited: a three cycle run must be allowed to finish."""
    settings.max_cycles = 0
    stub_agents.planner_script = [
        make_plan(make_task("t1", "one")),
        make_plan(make_task("t2", "two")),
        make_plan(make_task("t3", "three")),
    ]
    stub_agents.tester_script = [failing("red"), failing("red"), passing()]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert result.cycles == 3
    assert result.completed is True
    assert result.tasks_completed == 3


# ---------------------------------------------------------------------------
# guard against a broken loop
# ---------------------------------------------------------------------------


async def test_three_consecutive_planner_errors_stop_the_run(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A planner that always raises stops instead of looping forever."""
    settings.max_cycles = 0
    stub_agents.planner_script = [PlannerError("planner model did not return a usable plan")]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await orchestrator.run()

    assert result.reason.startswith("repeated cycle errors")
    assert "PlannerError" in result.reason
    assert result.cycles == 3
    assert result.completed is False
    assert len(events.of(EventKind.ERROR)) == 3


async def test_two_cycles_without_progress_stop_the_run(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """Two cycles that complete no task stop the run with ``no progress``."""
    settings.max_cycles = 50
    stub_agents.planner_script = [make_plan(make_task("t1", "doomed"))]

    def hook(task: TaskModel) -> str:
        raise RuntimeError(f"worker exploded on {task.id}")

    stub_agents.worker_hook = hook
    stub_agents.tester_script = [failing("nothing works")]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert result.reason == "no progress"
    assert result.cycles == 2
    assert result.tasks_completed == 0
    assert result.tasks_failed == 1


async def test_a_cycle_that_makes_progress_resets_the_stall_counter(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """Progress in an even numbered cycle keeps a run alive past the stall guard."""
    settings.max_cycles = 3
    stub_agents.planner_script = [
        make_plan(make_task("t1", "one")),
        make_plan(make_task("t2", "two")),
    ]

    def hook(task: TaskModel) -> str:
        if task.id == "t1":
            raise RuntimeError("first task failed")
        return "ok"

    stub_agents.worker_hook = hook
    stub_agents.tester_script = [failing("red"), failing("red")]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert result.reason == "cycle limit reached"
    assert result.cycles == 3


# ---------------------------------------------------------------------------
# the loop is bounded even when the planner keeps re-planning impossible work
# ---------------------------------------------------------------------------


async def test_green_tests_over_a_permanently_failing_task_cannot_loop_forever(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """The forever-loop repro terminates: a cycle that completes nothing is stalled.

    The planner always replans the same task, the worker always raises and the tester
    always reports success.  A green suite says nothing about work finishing, so it must
    not reset the stall counter; with ``max_cycles=0`` only the stall guard can stop this.
    """
    settings.max_cycles = 0  # unlimited: nothing but the guards can end the run
    stub_agents.planner_script = [make_plan(make_task("t1", "doomed"))]

    def hook(task: TaskModel) -> str:
        raise RuntimeError(f"worker exploded on {task.id}")

    stub_agents.worker_hook = hook
    stub_agents.tester_script = [passing("everything is green")]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await asyncio.wait_for(orchestrator.run(), timeout=10)

    assert result.completed is False
    assert result.reason == "no progress"
    assert result.cycles == 2
    assert result.tasks_completed == 0
    assert result.tasks_failed == 1
    assert result.tests_passed is True  # truthful: the suite really was green
    assert stub_agents.worker_calls == ["t1", "t1"]
    assert [event.cycle for event in events.of(EventKind.TESTS)] == [1, 2]


async def test_a_task_is_abandoned_after_max_task_attempts_and_not_run_again(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A task that used up its attempt budget is never dispatched again."""
    settings.max_cycles = 0
    settings.max_task_attempts = 2
    # The planner drops the healthy task after the first cycle, so from then on only the
    # doomed one is planned: the stall guard ends the run with it abandoned.
    stub_agents.planner_script = [
        make_plan(make_task("t_bad", "doomed"), make_task("t_ok", "fine")),
        make_plan(make_task("t_bad", "doomed")),
    ]

    def hook(task: TaskModel) -> str:
        if task.id == "t_bad":
            raise RuntimeError("worker exploded")
        return "ok"

    stub_agents.worker_hook = hook
    stub_agents.tester_script = [failing("red")]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await asyncio.wait_for(orchestrator.run(), timeout=10)

    assert stub_agents.worker_calls == ["t_bad", "t_ok", "t_bad"]
    doomed = next(task for task in orchestrator.tasks if task.id == "t_bad")
    assert doomed.status is TaskStatus.FAILED
    assert doomed.attempts == 2 == settings.max_task_attempts
    assert "abandoned after 2 attempts" in (doomed.error or "")
    abandoned = [
        event
        for event in events.of(EventKind.TASK_FAILED)
        if event.data.get("abandoned") is True
    ]
    assert [event.task_id for event in abandoned] == ["t_bad"]
    assert "abandoned after 2 attempts" in abandoned[0].message
    assert result.completed is False
    assert result.reason == "no progress"
    assert result.tasks_failed == 1


async def test_a_plan_made_only_of_abandoned_tasks_ends_the_run_cleanly(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """Nothing left to run: the run reports ``no progress`` instead of spinning."""
    settings.max_cycles = 0
    settings.max_task_attempts = 1
    stub_agents.planner_script = [make_plan(make_task("t1", "doomed"), make_task("t2", "also doomed"))]

    def hook(task: TaskModel) -> str:
        raise RuntimeError(f"worker exploded on {task.id}")

    stub_agents.worker_hook = hook
    stub_agents.tester_script = [passing("still green")]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await asyncio.wait_for(orchestrator.run(), timeout=10)

    assert result.reason == "no progress"
    assert result.cycles == 2
    assert result.completed is False
    assert result.tasks_completed == 0
    assert result.tasks_failed == 2
    assert stub_agents.worker_calls == ["t1", "t2"]  # never dispatched a second time
    assert all(
        "abandoned after 1 attempts" in (task.error or "")
        for task in orchestrator.tasks
        if task.status is TaskStatus.FAILED
    )
    assert events.kinds()[-1] is EventKind.DONE


async def test_a_final_cycle_that_only_skips_settled_work_still_completes(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A converging run is judged complete before the stall guard sees an empty cycle."""
    settings.max_cycles = 0
    stub_agents.planner_script = [
        make_plan(make_task("t1", "solid")),
        make_plan(make_task("t1", "solid")),
    ]
    stub_agents.tester_script = [failing("not yet"), passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await asyncio.wait_for(orchestrator.run(), timeout=10)

    assert result.completed is True
    assert result.reason == "goal complete"
    assert result.cycles == 2
    assert stub_agents.worker_calls == ["t1"]  # the second cycle skipped it
    assert [event.task_id for event in events.of(EventKind.TASK_SKIPPED)] == ["t1"]
    assert orchestrator.tasks[0].status is TaskStatus.SKIPPED
    assert result.tasks_completed == 1


# ---------------------------------------------------------------------------
# concurrency
# ---------------------------------------------------------------------------


async def test_worker_concurrency_never_exceeds_max_parallel_tasks(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """No more than ``max_parallel_tasks`` workers run at the same time."""
    settings.max_parallel_tasks = 2
    live = 0
    peak = 0

    async def hook(task: TaskModel) -> str:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        live -= 1
        return f"wrote {task.id}"

    stub_agents.worker_hook = hook
    stub_agents.planner_script = [make_plan(*(make_task(f"t{index}") for index in range(1, 6)))]
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert peak == 2
    assert sorted(stub_agents.worker_calls) == ["t1", "t2", "t3", "t4", "t5"]
    assert result.tasks_completed == 5
    assert result.completed is True


async def test_non_positive_max_parallel_tasks_still_runs_every_task(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """``max_parallel_tasks=0`` degrades to one task at a time instead of deadlocking."""
    settings.max_parallel_tasks = 0
    live = 0
    peak = 0

    async def hook(task: TaskModel) -> str:
        nonlocal live, peak
        live += 1
        peak = max(peak, live)
        await asyncio.sleep(0)
        live -= 1
        return f"wrote {task.id}"

    stub_agents.worker_hook = hook
    stub_agents.planner_script = [make_plan(*(make_task(f"t{index}") for index in range(1, 4)))]
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert peak == 1
    assert sorted(stub_agents.worker_calls) == ["t1", "t2", "t3"]
    assert result.tasks_completed == 3
    assert result.completed is True


# ---------------------------------------------------------------------------
# dependency waves
# ---------------------------------------------------------------------------


async def test_dependent_tasks_run_after_the_tasks_they_need(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """A task listed as depending on another only runs in a later wave."""
    stub_agents.planner_script = [
        make_plan(
            make_task("t1", "first"),
            make_task("t2", "second", dependencies=("t1",)),
            make_task("t3", "third", dependencies=("t2",)),
        )
    ]
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert stub_agents.worker_calls == ["t1", "t2", "t3"]
    assert result.completed is True


async def test_unknown_dependencies_are_ignored_and_the_task_still_runs(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A dependency on a task that does not exist is reported and then ignored."""
    stub_agents.planner_script = [make_plan(make_task("t1", "solo", dependencies=("ghost",)))]
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await orchestrator.run()

    assert stub_agents.worker_calls == ["t1"]
    assert result.completed is True
    assert any("unknown id" in message for message in events.messages_of(EventKind.LOG))


async def test_cyclic_dependencies_run_last_instead_of_hanging(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A dependency cycle neither hangs nor raises: both tasks still run."""
    stub_agents.planner_script = [
        make_plan(
            make_task("t3", "left", dependencies=("t4",)),
            make_task("t4", "right", dependencies=("t3",)),
        )
    ]
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await orchestrator.run()

    assert sorted(stub_agents.worker_calls) == ["t3", "t4"]
    assert result.tasks_completed == 2
    assert result.completed is True
    assert any("cyclic" in message for message in events.messages_of(EventKind.LOG))


async def test_unknown_and_cyclic_dependencies_in_one_plan_all_still_run(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """A malformed plan still runs every task exactly once."""
    stub_agents.planner_script = [
        make_plan(
            make_task("t1", "orphan", dependencies=("ghost",)),
            make_task("t2", "after orphan", dependencies=("t1",)),
            make_task("t3", "cycle a", dependencies=("t4",)),
            make_task("t4", "cycle b", dependencies=("t3",)),
        )
    ]
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert sorted(stub_agents.worker_calls) == ["t1", "t2", "t3", "t4"]
    assert result.completed is True


# ---------------------------------------------------------------------------
# already done work is not redone
# ---------------------------------------------------------------------------


async def test_a_task_completed_in_an_earlier_cycle_is_skipped_not_re_executed(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """Replanning the same description skips the finished task instead of running it."""
    stub_agents.planner_script = [
        make_plan(make_task("t1", "solid")),
        make_plan(make_task("t1", "solid"), make_task("t2", "extra")),
    ]
    stub_agents.tester_script = [failing("not yet"), passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await orchestrator.run()

    assert stub_agents.worker_calls == ["t1", "t2"]
    assert [event.task_id for event in events.of(EventKind.TASK_SKIPPED)] == ["t1"]
    assert orchestrator.tasks[0].status is TaskStatus.SKIPPED
    # ``tasks_completed`` counts terminal-done tasks, so the skipped-but-finished t1 counts
    # alongside the task the last cycle completed itself.
    assert result.tasks_completed == 2
    assert result.completed is True


async def test_a_task_that_failed_in_an_earlier_cycle_is_retried(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """A previously failed task is reset and executed again in the next cycle."""
    attempts: list[str] = []

    def hook(task: TaskModel) -> str:
        attempts.append(task.description)
        if len(attempts) == 1:
            raise RuntimeError("first attempt failed")
        return "second attempt worked"

    stub_agents.planner_script = [
        make_plan(make_task("t1", "flaky")),
        make_plan(make_task("t1", "flaky")),
    ]
    stub_agents.worker_hook = hook
    stub_agents.tester_script = [failing("red"), passing()]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert attempts == ["flaky", "flaky"]
    assert result.cycles == 2
    assert result.tasks_completed == 1
    assert result.tasks_failed == 0
    assert result.completed is True


async def test_completed_and_failed_tasks_are_skipped_and_retried_respectively(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """One cycle mixes both rules: the solid task is skipped, the flaky one retried."""
    attempts: dict[str, int] = {}

    def hook(task: TaskModel) -> str:
        attempts[task.description] = attempts.get(task.description, 0) + 1
        if task.description == "flaky" and attempts["flaky"] == 1:
            raise RuntimeError("flaky failed the first time")
        return f"ok {task.description}"

    stub_agents.planner_script = [
        make_plan(make_task("t1", "solid"), make_task("t2", "flaky")),
        make_plan(
            make_task("t1", "solid"),
            make_task("t2", "flaky"),
            make_task("t3", "extra"),
        ),
    ]
    stub_agents.worker_hook = hook
    stub_agents.tester_script = [failing("red"), passing()]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert attempts == {"solid": 1, "flaky": 2, "extra": 1}
    assert result.completed is True
    statuses = {task.description: task.status for task in orchestrator.tasks}
    assert statuses == {
        "solid": TaskStatus.SKIPPED,
        "flaky": TaskStatus.COMPLETED,
        "extra": TaskStatus.COMPLETED,
    }
    # COMPLETED and SKIPPED are both terminal-done states, so all three planned tasks count
    # as finished work: the count agrees with ``completed`` (FAILED is not done).
    assert result.tasks_completed == 3


# ---------------------------------------------------------------------------
# cooperative stop
# ---------------------------------------------------------------------------


async def test_stop_from_a_task_start_callback_ends_the_run_before_the_next_wave(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """``stop()`` from a ``TASK_START`` callback halts the run and skips later waves."""
    stub_agents.planner_script = [
        make_plan(
            make_task("t1", "wave one a"),
            make_task("t2", "wave one b"),
            make_task("t3", "wave two", dependencies=("t1",)),
        )
    ]
    stub_agents.tester_script = [passing()]
    collected: list[Event] = []
    orchestrator: Orchestrator

    async def on_event(event: Event) -> None:
        collected.append(event)
        if event.kind is EventKind.TASK_START and event.task_id == "t1":
            await orchestrator.stop()

    orchestrator = build(client=fake_client, settings=settings, on_event=on_event)

    result = await orchestrator.run()

    assert result.reason == "stop requested"
    assert result.completed is False
    assert result.cycles == 1
    assert "t3" not in stub_agents.worker_calls
    assert sorted(stub_agents.worker_calls) == ["t1", "t2"]
    assert orchestrator.stop_reason == "stop requested"
    kinds = [event.kind for event in collected]
    assert kinds[-1] is EventKind.DONE
    assert any("stop requested" in event.message for event in collected if event.kind is EventKind.LOG)


async def test_stop_emits_done_even_when_the_run_did_nothing(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """Stopping before the first cycle still ends with a ``DONE`` event."""
    settings.max_cycles = 5
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)
    await orchestrator.stop()

    result = await orchestrator.run()

    assert result.reason == "stop requested"
    assert stub_agents.worker_calls == []
    assert events.kinds()[-1] is EventKind.DONE
    assert events.of(EventKind.DONE)[0].data["reason"] == "stop requested"


# ---------------------------------------------------------------------------
# "no remaining work" is its own stop reason
# ---------------------------------------------------------------------------


async def test_an_empty_plan_after_passing_tests_reports_no_remaining_work(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """Once the tests are green, an empty plan ends the run as already satisfied."""
    stub_agents.planner_script = [make_plan(make_task("t1")), make_plan()]
    stub_agents.tester_script = [failing("red"), passing("12 passed")]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert result.completed is True
    assert result.reason == "planner reported no remaining work"
    assert result.cycles == 2


async def test_an_empty_plan_without_passing_tests_does_not_end_the_run(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """An empty plan alone is not success: the tests still have to pass."""
    settings.max_cycles = 1
    stub_agents.planner_script = [make_plan()]
    stub_agents.tester_script = [failing("still red")]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    assert result.completed is False
    assert result.reason == "cycle limit reached"


# ---------------------------------------------------------------------------
# the event stream
# ---------------------------------------------------------------------------


async def test_the_collector_sees_the_whole_lifecycle_with_done_last(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A successful run emits the full lifecycle and exactly one final ``DONE``."""
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    await orchestrator.run()

    kinds = events.kinds()
    for kind in (
        EventKind.CYCLE,
        EventKind.PLAN,
        EventKind.TASK_START,
        EventKind.TASK_DONE,
        EventKind.TESTS,
        EventKind.DONE,
    ):
        assert kind in kinds
    assert kinds[0] is EventKind.CYCLE
    assert kinds[-1] is EventKind.DONE
    assert kinds.count(EventKind.DONE) == 1
    assert [event.kind for event in orchestrator.events] == kinds


async def test_a_failed_task_is_announced_with_a_task_failed_event(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A worker that raises produces ``TASK_FAILED`` and does not end the run."""
    settings.max_cycles = 1
    stub_agents.planner_script = [make_plan(make_task("t1", "ok"), make_task("t2", "bad"))]

    def hook(task: TaskModel) -> str:
        if task.id == "t2":
            raise RuntimeError("worker exploded")
        return "fine"

    stub_agents.worker_hook = hook
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await orchestrator.run()

    failed = events.of(EventKind.TASK_FAILED)
    assert [event.task_id for event in failed] == ["t2"]
    assert "RuntimeError" in failed[0].data["summary"]
    assert result.completed is False
    assert result.tasks_failed == 1
    assert result.reason == "cycle limit reached"
    assert events.kinds()[-1] is EventKind.DONE


async def test_every_event_payload_is_json_serializable(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """``to_dict()`` output survives ``json.dumps`` for every emitted event."""
    settings.max_cycles = 1
    stub_agents.planner_script = [make_plan(make_task("t1"), make_task("t2"))]

    def hook(task: TaskModel) -> str:
        if task.id == "t2":
            raise RuntimeError("boom")
        return "fine"

    stub_agents.worker_hook = hook
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    await orchestrator.run()

    assert events.events
    for event in events.events:
        payload = json.loads(json.dumps(event.to_dict()))
        assert payload["kind"] == event.kind.value
        assert isinstance(payload["data"], dict)
        assert payload["message"] == event.message


async def test_a_passing_tests_event_carries_the_flag_and_summary(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A green suite is announced with ``passed=True`` and its summary."""
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    stub_agents.tester_script = [passing("7 passed")]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    await orchestrator.run()

    tests = events.of(EventKind.TESTS)[-1]
    assert tests.data["passed"] is True
    assert tests.data["summary"] == "7 passed"
    assert tests.data["command"] == "pytest -q"
    assert tests.data["failures"] == 0


async def test_a_failing_tests_event_carries_the_flag_and_failure_count(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """A red suite is announced with ``passed=False`` and its failures."""
    settings.max_cycles = 1
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    stub_agents.tester_script = [failing("a", "b", summary="2 failed")]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    await orchestrator.run()

    tests = events.of(EventKind.TESTS)[-1]
    assert tests.data["passed"] is False
    assert tests.data["summary"] == "2 failed"
    assert tests.data["failures"] == 2


async def test_the_done_event_carries_the_run_result(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """The final ``DONE`` payload describes the run that just finished."""
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    stub_agents.tester_script = [passing()]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await orchestrator.run()

    done = events.of(EventKind.DONE)[0]
    assert done.data["completed"] is result.completed
    assert done.data["reason"] == result.reason
    assert done.data["cycles"] == result.cycles
    assert done.data["tasks_completed"] == result.tasks_completed
    assert done.data["tests_passed"] is True


async def test_an_exploding_tester_is_reported_and_the_run_still_returns(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
    events: EventCollector,
) -> None:
    """An ordinary agent failure is surfaced as an event, never raised out of ``run()``."""
    settings.max_cycles = 1
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    stub_agents.tester_script = [RuntimeError("test runner vanished")]
    orchestrator = build(client=fake_client, settings=settings, on_event=events)

    result = await orchestrator.run()

    errors = events.of(EventKind.ERROR)
    assert len(errors) == 1
    assert "test runner vanished" in errors[0].data["summary"]
    assert result.completed is False
    assert result.reason == "cycle limit reached"


# ---------------------------------------------------------------------------
# BaseExceptions are not swallowed
# ---------------------------------------------------------------------------


async def test_keyboard_interrupt_propagates_out_of_run(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """A Ctrl-C is not an ordinary failure: it must reach the caller."""
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    stub_agents.tester_script = [KeyboardInterrupt()]
    orchestrator = build(client=fake_client, settings=settings)

    with pytest.raises(KeyboardInterrupt):
        await orchestrator.run()


async def test_cancellation_propagates_out_of_run(
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """``asyncio.CancelledError`` must not be mistaken for an ordinary cycle failure."""
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    stub_agents.tester_script = [asyncio.CancelledError()]
    orchestrator = build(client=fake_client, settings=settings)

    with pytest.raises(asyncio.CancelledError):
        await orchestrator.run()


# ---------------------------------------------------------------------------
# console rendering must never interpret untrusted text as markup
# ---------------------------------------------------------------------------


async def test_rich_markup_in_task_text_is_rendered_literally(
    monkeypatch: pytest.MonkeyPatch,
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """Square brackets coming from a task id or a worker result are printed as text."""
    settings.max_cycles = 1  # the broken task would otherwise be retried forever
    buffer = io.StringIO()
    monkeypatch.setattr(
        orchestrator_module,
        "console",
        Console(file=buffer, width=200, no_color=True, highlight=False),
    )
    stub_agents.planner_script = [
        make_plan(
            make_task("t1[bold red]", "patch [bold red]widget.py"),
            make_task("t2", "explode [bold red]"),
        )
    ]

    def hook(task: TaskModel) -> str:
        if task.id == "t2":
            raise RuntimeError("[/] RuntimeError: [bold red] cannot open widget")
        return "patched [bold red]widget.py [/]"

    stub_agents.worker_hook = hook
    stub_agents.tester_script = [passing("[bold red]5 passed [/]")]
    orchestrator = build(client=fake_client, settings=settings)

    result = await orchestrator.run()

    printed = buffer.getvalue()
    assert result.completed is False
    assert "[bold red]" in printed
    assert "[/]" in printed
    assert "cannot open widget" in printed


async def test_markup_in_the_goal_and_reason_is_rendered_literally(
    monkeypatch: pytest.MonkeyPatch,
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """A goal containing markup does not abort the run with a markup error."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        orchestrator_module,
        "console",
        Console(file=buffer, width=200, no_color=True, highlight=False),
    )
    settings.max_cycles = 1
    stub_agents.planner_script = [make_plan(make_task("t1", "write [bold]code[/bold]"))]
    stub_agents.tester_script = [passing()]
    orchestrator = build(
        "ship [bold red]/etc/passwd[/bold red] safely",
        client=fake_client,
        settings=settings,
    )

    result = await orchestrator.run()

    printed = buffer.getvalue()
    assert result.completed is True
    assert "[bold red]" in printed
    assert "[bold]" in printed
    assert "ship [bold red]/etc/passwd[/bold red] safely" in printed


# ---------------------------------------------------------------------------
# the summary panel belongs to whoever is rendering
# ---------------------------------------------------------------------------


async def test_the_summary_panel_is_only_printed_without_an_event_callback(
    monkeypatch: pytest.MonkeyPatch,
    stub_agents: StubAgents,
    fake_client: FakeClient,
    settings: Settings,
) -> None:
    """With an ``on_event`` consumer the consumer renders the summary; the CLI would print it twice."""
    buffer = io.StringIO()
    monkeypatch.setattr(
        orchestrator_module,
        "console",
        Console(file=buffer, width=200, no_color=True, highlight=False),
    )
    stub_agents.planner_script = [make_plan(make_task("t1"))]
    stub_agents.tester_script = [passing()]

    await build(client=fake_client, settings=settings, on_event=EventCollector()).run()

    assert "Run summary" not in buffer.getvalue()

    buffer.seek(0)
    buffer.truncate(0)
    await build(client=fake_client, settings=settings).run()

    assert buffer.getvalue().count("Run summary") == 1