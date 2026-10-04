"""Tests for :mod:`furrow.core.orchestrator`: the plan/execute/test loop.

Every agent is replaced with a deterministic fake, so the loop can be driven
through any sequence of plans, worker outcomes and test verdicts without an LLM.
The behaviours locked down here are the ones that previously regressed:
``_is_done`` returning a vacuous ``0 >= 0``, ``_get_tasks`` handing back a fresh
list every call, ``original_goal`` being overwritten by the repair preamble, and
the dependency scheduler hanging on a cycle.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.events import EventType, FurrowEvent, RunResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMError

from .conftest import FakeLLMClient

# --------------------------------------------------------------------------- #
# test doubles for the three agents
# --------------------------------------------------------------------------- #


class ScriptedPlanner:
    """A planner that returns queued plans (or raises queued exceptions)."""

    def __init__(self, plans: list[Any]) -> None:
        """Store the script.

        Args:
            plans: Items consumed one per ``plan`` call. A :class:`Plan` is
                returned, an exception instance is raised, and a callable is
                invoked with the goal and must return a :class:`Plan`.
        """
        self.plans = list(plans)
        self.goals: list[str] = []
        self.instances = 0

    async def plan(self, goal: str) -> Plan:
        """Produce the next queued outcome.

        Args:
            goal: The goal the orchestrator is asking about.

        Returns:
            The next queued :class:`Plan`.

        Raises:
            BaseException: Whatever exception instance was queued.
        """
        self.goals.append(goal)
        if not self.plans:
            raise AssertionError(f"ScriptedPlanner exhausted after {self.goals!r}")
        item = self.plans.pop(0)
        if isinstance(item, BaseException):
            raise item
        if callable(item):
            return item(goal)
        return item


class ScriptedWorker:
    """A worker whose result and concurrency behaviour are scriptable."""

    def __init__(
        self,
        task: TaskModel,
        result: str = "done",
        error: BaseException | None = None,
        delay: float = 0.0,
        clock: _ConcurrencyClock | None = None,
    ) -> None:
        """Configure one worker invocation.

        Args:
            task: The task handed over by the orchestrator.
            result: Text returned by ``run`` on success.
            error: Exception raised by ``run``.
            delay: Seconds slept inside ``run``, to widen the concurrency window.
            clock: Shared counter used by the concurrency tests.
        """
        self.task = task
        self.result = result
        self.error = error
        self.delay = delay
        self.clock = clock
        self.files_written = [f"{task.id}.py"]
        self.errors: list[str] = []

    async def run(self) -> str:
        """Simulate the worker body.

        Returns:
            ``self.result``.

        Raises:
            BaseException: ``self.error`` when one was configured.
        """
        if self.clock is not None:
            self.clock.enter(self.task.id)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error is not None:
                raise self.error
            return self.result
        finally:
            if self.clock is not None:
                self.clock.exit(self.task.id)


class _ConcurrencyClock:
    """Records worker overlap so peak parallelism can be asserted on."""

    def __init__(self) -> None:
        """Start with an empty observation record."""
        self.active = 0
        self.peak = 0
        self.log: list[str] = []
        self.started: set[str] = set()
        self.finished: set[str] = set()

    def enter(self, task_id: str) -> None:
        """Record a worker starting."""
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.log.append(f"start:{task_id}")
        self.started.add(task_id)

    def exit(self, task_id: str) -> None:
        """Record a worker finishing."""
        self.active -= 1
        self.log.append(f"end:{task_id}")
        self.finished.add(task_id)

    def index_of(self, needle: str) -> int:
        """Return where ``needle`` first appears in the log.

        Args:
            needle: The log entry to find.

        Returns:
            The zero-based index of the first match.
        """
        return self.log.index(needle)


class ScriptedTester:
    """A tester returning queued :class:`TestResult` values."""

    def __init__(self, results: list[Any]) -> None:
        """Store the script.

        Args:
            results: Items consumed one per ``run`` call; exceptions are raised.
        """
        self.results = list(results)
        self.calls: list[tuple[str, list[str]]] = []

    async def run(self, goal: str, tasks: list[TaskModel] | None = None) -> TestResult:
        """Produce the next queued verdict.

        Args:
            goal: The goal under test.
            tasks: The task registry snapshot.

        Returns:
            The next queued :class:`TestResult`.

        Raises:
            BaseException: Whatever exception instance was queued.
        """
        self.calls.append((goal, [task.id for task in tasks or []]))
        if not self.results:
            raise AssertionError(f"ScriptedTester exhausted after {self.calls!r}")
        item = self.results.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _plan(*specs: tuple[str, list[str]]) -> Plan:
    """Build a plan from ``(id, dependencies)`` pairs.

    Args:
        *specs: One ``(task_id, dependencies)`` tuple per task.

    Returns:
        The plan, with every task ``PENDING``.
    """
    return Plan(
        tasks=[
            TaskModel(id=task_id, description=f"do {task_id}", dependencies=deps)
            for task_id, deps in specs
        ],
        rationale="scripted",
    )


def _passing(summary: str = "suite green") -> TestResult:
    """Build a passing :class:`TestResult`.

    Args:
        summary: The summary line.

    Returns:
        The result.
    """
    return TestResult(passed=True, summary=summary)


def _failing(summary: str = "suite red") -> TestResult:
    """Build a failing :class:`TestResult`.

    Args:
        summary: The summary line.

    Returns:
        The result with one synthetic failure.
    """
    return TestResult(passed=False, summary=summary, failures=["tests/test_a.py::test_x FAILED"])


def install_agents(
    monkeypatch: pytest.MonkeyPatch,
    planner: ScriptedPlanner,
    tester: ScriptedTester,
    worker_factory: Any = None,
) -> None:
    """Patch the three agent names the orchestrator resolves at construction.

    Args:
        monkeypatch: Pytest's patcher.
        planner: The planner double.
        tester: The tester double.
        worker_factory: Called as ``worker_factory(task=task, client=..., settings=...)``
            to build each worker. Defaults to a :class:`ScriptedWorker` that
            succeeds immediately.
    """
    if worker_factory is None:

        def worker_factory(task: TaskModel, client: Any, settings: Any) -> ScriptedWorker:
            return ScriptedWorker(task)

    monkeypatch.setattr(
        orchestrator_module, "PlannerAgent", lambda client=None, settings=None: planner
    )
    monkeypatch.setattr(
        orchestrator_module, "TesterAgent", lambda client=None, settings=None: tester
    )
    monkeypatch.setattr(
        orchestrator_module, "WorkerAgent", lambda task, client=None, settings=None: worker_factory(
            task=task, client=client, settings=settings
        )
    )


def make_orchestrator(
    settings: Settings,
    on_event: Any = None,
    **kwargs: Any,
) -> Orchestrator:
    """Build an orchestrator wired to a fake client and explicit settings.

    Args:
        settings: The settings every component shares.
        on_event: Optional lifecycle listener.
        **kwargs: Extra :class:`Orchestrator` keyword arguments.

    Returns:
        The orchestrator.
    """
    return Orchestrator(
        goal=kwargs.pop("goal", "make it work"),
        client=FakeLLMClient(settings=settings),
        settings=settings,
        on_event=on_event,
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# _is_done: the vacuous-truthiness regression
# --------------------------------------------------------------------------- #


def test_is_done_is_false_for_an_empty_registry(settings_factory: Any) -> None:
    """The old ``0 >= 0`` body reported every run as finished on cycle one."""
    orchestrator = make_orchestrator(settings_factory())

    assert orchestrator.tasks == []
    assert orchestrator._is_done() is False


def test_is_done_is_false_for_mixed_statuses(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory())
    orchestrator.tasks = [
        TaskModel(id="1", description="d", status=TaskStatus.COMPLETED),
        TaskModel(id="2", description="d", status=TaskStatus.RUNNING),
    ]

    assert orchestrator._is_done() is False


def test_is_done_is_false_when_every_task_completed_but_the_registry_is_empty(
    settings_factory: Any,
) -> None:
    orchestrator = make_orchestrator(settings_factory())

    assert all(task.status is TaskStatus.COMPLETED for task in orchestrator.tasks) is True
    assert orchestrator._is_done() is False


def test_is_done_is_false_when_one_task_failed(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory())
    orchestrator.tasks = [
        TaskModel(id="1", description="d", status=TaskStatus.COMPLETED),
        TaskModel(id="2", description="d", status=TaskStatus.FAILED),
    ]

    assert orchestrator._is_done() is False


def test_is_done_is_false_when_one_task_is_pending(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory())
    orchestrator.tasks = [
        TaskModel(id="1", description="d", status=TaskStatus.COMPLETED),
        TaskModel(id="2", description="d", status=TaskStatus.PENDING),
    ]

    assert orchestrator._is_done() is False


def test_is_done_is_true_when_every_task_completed(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory())
    orchestrator.tasks = [
        TaskModel(id="1", description="d", status=TaskStatus.COMPLETED),
        TaskModel(id="2", description="d", status=TaskStatus.COMPLETED),
    ]

    assert orchestrator._is_done() is True


# --------------------------------------------------------------------------- #
# _get_tasks: the fresh-list regression
# --------------------------------------------------------------------------- #


def test_get_tasks_returns_the_live_registry(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory())

    assert orchestrator._get_tasks() is orchestrator.tasks


def test_get_tasks_accumulates_across_calls(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory())
    orchestrator.tasks.append(TaskModel(id="1", description="d"))

    first = orchestrator._get_tasks()
    orchestrator.tasks.append(TaskModel(id="2", description="d"))
    second = orchestrator._get_tasks()

    assert first is second
    assert [task.id for task in second] == ["1", "2"]


async def test_get_tasks_accumulates_across_cycles(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any, event_collector: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", [])), _plan(("2", []))])
    tester = ScriptedTester([_failing(), _passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=2)
    events, on_event = event_collector

    orchestrator = make_orchestrator(settings, on_event=on_event)
    await orchestrator.run()

    assert [task.id for task in orchestrator._get_tasks()] == ["1", "2"]


# --------------------------------------------------------------------------- #
# goal integrity
# --------------------------------------------------------------------------- #


def test_original_goal_and_goal_alias_start_equal(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory(), goal="ship the parser")

    assert orchestrator.original_goal == "ship the parser"
    assert orchestrator.current_goal == "ship the parser"
    assert orchestrator.goal == "ship the parser"


def test_setting_the_goal_alias_leaves_original_goal_alone(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory(), goal="ship the parser")

    orchestrator.goal = "something else"

    assert orchestrator.current_goal == "something else"
    assert orchestrator.original_goal == "ship the parser"


async def test_original_goal_survives_a_failing_test_run(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    goal = "add a health endpoint"
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_failing("2 tests failed")])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    orchestrator = make_orchestrator(settings, goal=goal)

    await orchestrator.run()

    assert orchestrator.original_goal == goal
    assert orchestrator.current_goal.startswith(goal)
    assert "2 tests failed" in orchestrator.current_goal
    assert "tests/test_a.py::test_x FAILED" in orchestrator.current_goal


async def test_repair_preamble_is_appended_not_substituted(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    goal = "add a health endpoint"
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    orchestrator = make_orchestrator(settings, goal=goal)

    await orchestrator.run()

    assert goal in orchestrator.current_goal
    assert orchestrator.current_goal != goal
    assert len(orchestrator.current_goal) > len(goal)


async def test_a_green_suite_restores_the_original_goal(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    goal = "add a health endpoint"
    planner = ScriptedPlanner([_plan(("1", [])), _plan(("2", []))])
    tester = ScriptedTester([_failing(), _passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=2)
    orchestrator = make_orchestrator(settings, goal=goal)

    await orchestrator.run()

    assert orchestrator.original_goal == goal
    assert orchestrator.current_goal == goal


# --------------------------------------------------------------------------- #
# cycle budget and stall guards
# --------------------------------------------------------------------------- #


async def test_max_cycles_is_honoured(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    orchestrator = make_orchestrator(settings)

    result = await orchestrator.run()

    assert result.cycles == 1
    assert orchestrator.cycles == 1
    assert result.completed is False
    assert "max_cycles" in result.stopped_reason


async def test_unlimited_cycles_still_terminate_on_an_always_failing_suite(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []))] * 20)
    tester = ScriptedTester([_failing()] * 20)
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=0)
    orchestrator = make_orchestrator(settings)

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert result.completed is False
    assert result.stopped_reason
    assert result.cycles <= Orchestrator.MAX_UNPRODUCTIVE_CYCLES + 2
    assert "stalled" in result.stopped_reason or "repair" in result.stopped_reason


async def test_repair_guard_bounds_a_repair_loop(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """A repair loop that keeps making new tasks stops after the attempt cap."""
    planner = ScriptedPlanner([_plan((str(index), [])) for index in range(1, 30)])
    tester = ScriptedTester([_failing()] * 29)
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=0)
    orchestrator = make_orchestrator(settings)

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert result.completed is False
    assert result.cycles <= Orchestrator.MAX_REPAIR_ATTEMPTS + 1


async def test_stalled_loop_is_bounded_by_the_documented_constant() -> None:
    assert Orchestrator.MAX_UNPRODUCTIVE_CYCLES == 3
    assert Orchestrator.MAX_REPAIR_ATTEMPTS == 3


# --------------------------------------------------------------------------- #
# dependency scheduling
# --------------------------------------------------------------------------- #


async def test_dependent_task_starts_after_its_dependency_completes(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    clock = _ConcurrencyClock()

    def worker_factory(task: TaskModel, client: Any, settings: Any) -> ScriptedWorker:
        return ScriptedWorker(task, delay=0.01, clock=clock)

    planner = ScriptedPlanner([_plan(("a", []), ("b", ["a"]), ("c", ["b"]))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester, worker_factory)
    settings = settings_factory(max_parallel_tasks=5, max_cycles=1)

    orchestrator = make_orchestrator(settings)
    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert result.completed is True
    assert clock.index_of("start:b") > clock.index_of("end:a")
    assert clock.index_of("start:c") > clock.index_of("end:b")


async def test_max_parallel_tasks_caps_observed_concurrency(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    clock = _ConcurrencyClock()

    def worker_factory(task: TaskModel, client: Any, settings: Any) -> ScriptedWorker:
        return ScriptedWorker(task, delay=0.02, clock=clock)

    specs = tuple((str(index), []) for index in range(1, 9))
    planner = ScriptedPlanner([_plan(*specs)])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester, worker_factory)
    settings = settings_factory(max_parallel_tasks=2, max_cycles=1)

    orchestrator = make_orchestrator(settings)
    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert result.tasks_completed == 8
    assert clock.peak <= 2
    assert clock.peak == 2


async def test_independent_tasks_do_run_concurrently(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    clock = _ConcurrencyClock()

    def worker_factory(task: TaskModel, client: Any, settings: Any) -> ScriptedWorker:
        return ScriptedWorker(task, delay=0.02, clock=clock)

    planner = ScriptedPlanner([_plan(("a", []), ("b", []), ("c", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester, worker_factory)
    settings = settings_factory(max_parallel_tasks=3, max_cycles=1)

    await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert clock.peak == 3


async def test_a_failed_dependency_skips_its_dependents(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    def worker_factory(task: TaskModel, client: Any, settings: Any) -> ScriptedWorker:
        error = RuntimeError("worker exploded") if task.id == "a" else None
        return ScriptedWorker(task, error=error)

    planner = ScriptedPlanner([_plan(("a", []), ("b", ["a"]), ("c", ["b"]))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester, worker_factory)
    settings = settings_factory(max_parallel_tasks=5, max_cycles=1)
    events: list[FurrowEvent] = []
    orchestrator = make_orchestrator(settings, on_event=events.append)

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    by_id = {task.id: task for task in orchestrator.tasks}
    assert by_id["a"].status is TaskStatus.FAILED
    assert by_id["b"].status is TaskStatus.SKIPPED
    assert by_id["c"].status is TaskStatus.SKIPPED
    assert "a" in by_id["b"].result
    assert result.tasks_failed == 1
    skipped = [event for event in events if event.type is EventType.TASK_SKIPPED]
    assert len(skipped) == 2


async def test_a_skipped_dependency_also_skips_its_dependents(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("a", ["b"]), ("b", ["a"]), ("c", ["a"]))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    orchestrator = make_orchestrator(settings)

    await asyncio.wait_for(orchestrator.run(), timeout=30)

    by_id = {task.id: task for task in orchestrator.tasks}
    assert by_id["a"].status is TaskStatus.SKIPPED
    assert by_id["b"].status is TaskStatus.SKIPPED
    assert by_id["c"].status is TaskStatus.SKIPPED


async def test_a_dependency_cycle_terminates_with_everything_skipped(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """A cycle in the dependency graph must not deadlock the scheduler."""
    planner = ScriptedPlanner([_plan(("a", ["b"]), ("b", ["a"]), ("c", ["a"]), ("d", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    orchestrator = make_orchestrator(settings)

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    by_id = {task.id: task for task in orchestrator.tasks}
    assert by_id["a"].status is TaskStatus.SKIPPED
    assert by_id["b"].status is TaskStatus.SKIPPED
    assert by_id["c"].status is TaskStatus.SKIPPED
    assert by_id["d"].status is TaskStatus.COMPLETED
    assert "cycle" in (by_id["a"].result or "")
    assert result.completed is False


async def test_an_unknown_dependency_is_reported_and_the_task_still_runs(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("a", ["ghost"]))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events: list[FurrowEvent] = []
    orchestrator = make_orchestrator(settings, on_event=events.append)

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert orchestrator.tasks[0].status is TaskStatus.COMPLETED
    assert result.completed is True
    errors = [event for event in events if event.type is EventType.ERROR]
    assert errors and "ghost" in errors[0].message


# --------------------------------------------------------------------------- #
# gather(return_exceptions=True)
# --------------------------------------------------------------------------- #


async def test_a_worker_that_cannot_be_built_does_not_block_its_siblings(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    def worker_factory(task: TaskModel, client: Any, settings: Any) -> ScriptedWorker:
        if task.id == "bad":
            raise RuntimeError("worker could not be constructed")
        return ScriptedWorker(task)

    planner = ScriptedPlanner([_plan(("bad", []), ("good1", []), ("good2", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester, worker_factory)
    settings = settings_factory(max_parallel_tasks=3, max_cycles=1)
    orchestrator = make_orchestrator(settings)

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    by_id = {task.id: task for task in orchestrator.tasks}
    assert by_id["bad"].status is TaskStatus.FAILED
    assert by_id["good1"].status is TaskStatus.COMPLETED
    assert by_id["good2"].status is TaskStatus.COMPLETED
    assert result.tasks_completed == 2
    assert result.tasks_failed == 1


async def test_a_raising_worker_sibling_still_completes(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    def worker_factory(task: TaskModel, client: Any, settings: Any) -> ScriptedWorker:
        error = LLMError("provider down") if task.id == "bad" else None
        return ScriptedWorker(task, error=error)

    planner = ScriptedPlanner([_plan(("bad", []), ("good", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester, worker_factory)
    settings = settings_factory(max_parallel_tasks=2, max_cycles=1)
    orchestrator = make_orchestrator(settings)

    await asyncio.wait_for(orchestrator.run(), timeout=30)

    by_id = {task.id: task for task in orchestrator.tasks}
    assert by_id["bad"].status is TaskStatus.FAILED
    assert by_id["good"].status is TaskStatus.COMPLETED


# --------------------------------------------------------------------------- #
# resilience
# --------------------------------------------------------------------------- #


async def test_a_planner_value_error_is_absorbed_and_the_loop_continues(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([ValueError("planner returned no tasks"), _plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=2)
    orchestrator = make_orchestrator(settings)

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert result.cycles_failed == 1
    assert result.last_error is not None
    assert "ValueError" in result.last_error
    assert result.cycles == 2
    assert result.completed is True


async def test_an_llm_error_from_the_planner_is_absorbed(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([LLMError("provider exploded"), _plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=2)

    result = await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert result.cycles_failed == 1
    assert "LLMError" in (result.last_error or "")
    assert result.completed is True


async def test_an_llm_error_from_the_tester_is_absorbed(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", [])), _plan(("2", []))])
    tester = ScriptedTester([LLMError("judge exploded"), _passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=2)

    result = await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert result.cycles_failed == 1
    assert "LLMError" in (result.last_error or "")
    assert result.tasks_completed == 2
    assert result.completed is True


async def test_a_tester_that_cannot_be_built_does_not_complete_the_run(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    class _ExplodingTester:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            raise RuntimeError("tester could not be constructed")

    planner = ScriptedPlanner([_plan(("1", []))])
    install_agents(monkeypatch, planner, ScriptedTester([]))
    monkeypatch.setattr(
        orchestrator_module, "TesterAgent", lambda client=None, settings=None: _ExplodingTester()
    )
    settings = settings_factory(max_cycles=1)

    result = await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert result.completed is False
    assert result.cycles == 1


async def test_a_fully_failing_planner_still_terminates(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([ValueError("no tasks")] * 20)
    tester = ScriptedTester([_passing()] * 20)
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=0)

    result = await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert result.completed is False
    assert result.stopped_reason
    assert result.cycles >= 1


# --------------------------------------------------------------------------- #
# RunResult semantics
# --------------------------------------------------------------------------- #


async def test_completed_is_false_when_the_suite_is_red_despite_every_task_passing(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """This is what makes the CLI exit code correct."""
    planner = ScriptedPlanner([_plan(("1", []), ("2", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)

    result = await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert result.tasks_completed == 2
    assert result.tasks_failed == 0
    assert result.completed is False


async def test_completed_is_true_when_every_task_passes_and_the_suite_is_green(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []), ("2", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)

    result = await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert result.completed is True
    assert result.tasks_completed == 2
    assert "2 planned tasks completed" in result.stopped_reason


async def test_a_green_suite_with_an_empty_plan_can_complete(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """An empty plan alone is never 'done', but a green suite is not blocked on it."""
    planner = ScriptedPlanner([_plan()] * 4)
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=3)

    result = await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert result.tasks_completed == 0
    assert result.completed is True


async def test_stopped_reason_is_always_populated(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)

    result = await asyncio.wait_for(make_orchestrator(settings).run(), timeout=30)

    assert isinstance(result, RunResult)
    assert result.stopped_reason.strip()


async def test_replanned_tasks_are_reused_by_id_and_keep_their_status(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", [])), _plan(("1", []))])
    tester = ScriptedTester([_failing(), _passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=2)
    orchestrator = make_orchestrator(settings)

    await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert len(orchestrator.tasks) == 1
    assert orchestrator.tasks[0].status is TaskStatus.COMPLETED


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #


async def test_on_event_receives_the_full_lifecycle_sequence(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any, event_collector: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events, on_event = event_collector

    result = await asyncio.wait_for(
        make_orchestrator(settings, on_event=on_event).run(), timeout=30
    )

    sequence = [event.type for event in events]
    assert sequence[0] is EventType.RUN_START
    assert sequence[-1] is EventType.RUN_END
    for expected in (
        EventType.CYCLE_START,
        EventType.PLAN,
        EventType.TASK_START,
        EventType.TASK_COMPLETE,
        EventType.TEST_START,
        EventType.TEST_RESULT,
        EventType.CYCLE_END,
    ):
        assert expected in sequence
    assert sequence.index(EventType.CYCLE_START) < sequence.index(EventType.PLAN)
    assert sequence.index(EventType.TASK_COMPLETE) < sequence.index(EventType.TEST_START)
    assert sequence.index(EventType.TEST_RESULT) < sequence.index(EventType.CYCLE_END)
    # ``RUN_END`` once shipped its ``RunResult`` under ``data["data"]`` because
    # ``_emit`` takes ``**data`` and was handed a literal ``data=`` keyword. That
    # nesting was a bug, and it is fixed: the web server forwards ``event``s
    # verbatim, so every payload now sits flat beside its siblings and consumers
    # read ``event.data["completed"]`` whichever type they are handling.
    assert events[-1].data["completed"] is result.completed
    assert events[-1].data["stopped_reason"] == result.stopped_reason
    assert events[-1].data["cycles"] == result.cycles
    assert events[-1].data["tasks_completed"] == result.tasks_completed


async def test_test_result_events_flatten_their_payload(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """``TEST_RESULT`` and ``RUN_END`` now share one flat payload contract."""
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events: list[FurrowEvent] = []

    await asyncio.wait_for(
        make_orchestrator(settings, on_event=events.append).run(), timeout=30
    )

    verdict = next(event for event in events if event.type is EventType.TEST_RESULT)
    assert verdict.data["passed"] is False
    assert verdict.data["failures"]


async def test_events_carry_task_identity_and_are_json_serialisable(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_failing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events: list[FurrowEvent] = []

    await asyncio.wait_for(
        make_orchestrator(settings, on_event=events.append).run(), timeout=30
    )

    started = next(e for e in events if e.type is EventType.TASK_START)
    assert started.task_id == "1"
    assert started.cycle == 1
    assert started.model_dump(mode="json")["type"] == "task_start"
    failed = next(e for e in events if e.type is EventType.TEST_RESULT)
    assert failed.data["passed"] is False


async def test_an_on_event_listener_that_raises_is_ignored(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    calls = []

    def listener(event: FurrowEvent) -> None:
        calls.append(event)
        raise RuntimeError("listener is broken")

    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)

    result = await asyncio.wait_for(
        make_orchestrator(settings, on_event=listener).run(), timeout=30
    )

    assert calls
    assert result.completed is True


async def test_an_on_event_listener_that_raises_mid_run_is_ignored(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    seen: list[EventType] = []

    def listener(event: FurrowEvent) -> None:
        seen.append(event.type)
        if event.type is EventType.TASK_COMPLETE:
            raise RuntimeError("listener is broken")

    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)

    result = await asyncio.wait_for(
        make_orchestrator(settings, on_event=listener).run(), timeout=30
    )

    assert EventType.TASK_COMPLETE in seen
    assert EventType.RUN_END in seen
    assert result.completed is True


# --------------------------------------------------------------------------- #
# the flat payload contract every event type shares
# --------------------------------------------------------------------------- #


async def _every_event_type_in_one_run(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> list[FurrowEvent]:
    """Drive one cycle arranged to emit every :class:`EventType` there is.

    A raising worker, a dependent of the failed task and a task naming an
    unknown dependency are packed into a single plan, so the failure, skip and
    error events join the ordinary ones and the returned list exercises all
    twelve types.

    Args:
        monkeypatch: Pytest's patcher.
        settings: Settings shared by every component of the run.

    Returns:
        Every event the run emitted, in order.
    """

    def worker_factory(task: TaskModel, client: Any, settings: Any) -> ScriptedWorker:
        error = RuntimeError("worker exploded") if task.id == "a" else None
        return ScriptedWorker(task, error=error)

    planner = ScriptedPlanner([_plan(("a", []), ("b", ["a"]), ("c", ["ghost"]))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester, worker_factory)
    events: list[FurrowEvent] = []

    await asyncio.wait_for(make_orchestrator(settings, on_event=events.append).run(), timeout=30)

    return events


async def test_no_event_nests_its_payload_under_a_data_key(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """``_emit`` takes ``**data``, so nothing may ever land in ``data["data"]``.

    ``run_end`` used to be the lone exception, which forced consumers to reach
    through one extra level while every other type was flat.
    """
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events: list[FurrowEvent] = []

    await asyncio.wait_for(make_orchestrator(settings, on_event=events.append).run(), timeout=30)

    assert events
    assert [event.type for event in events if "data" in event.data] == []


async def test_run_end_payload_is_exactly_the_flat_run_result(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """``run_end`` carries the seven ``RunResult`` fields and nothing else."""
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events: list[FurrowEvent] = []

    result = await asyncio.wait_for(
        make_orchestrator(settings, on_event=events.append).run(), timeout=30
    )

    assert set(events[-1].data) == {
        "cycles",
        "completed",
        "tasks_completed",
        "tasks_failed",
        "cycles_failed",
        "last_error",
        "stopped_reason",
    }
    assert events[-1].data == result.model_dump(mode="json")


async def test_every_event_type_payload_is_flat_and_json_serialisable(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """``model_dump(mode="json")`` is the web-socket wire format and must not raise."""
    events = await _every_event_type_in_one_run(monkeypatch, settings_factory(max_cycles=1))

    emitted = {event.type for event in events}
    assert emitted == set(EventType)
    for event in events:
        assert "data" not in event.data, event.type
        assert json.loads(json.dumps(event.model_dump(mode="json")))["type"] == event.type.value


# --------------------------------------------------------------------------- #
# the _emit 'data' guard
# --------------------------------------------------------------------------- #


def test_emit_rejects_a_literal_data_keyword(settings_factory: Any) -> None:
    """A nested payload is a call-site bug, so it fails loudly instead of shipping."""
    events: list[FurrowEvent] = []
    orchestrator = make_orchestrator(settings_factory(), on_event=events.append)

    with pytest.raises(TypeError) as excinfo:
        orchestrator._emit(EventType.ERROR, message="m", data={"x": 1})

    message = str(excinfo.value)
    assert "data" in message
    assert "splat" in message
    assert events == []


def test_emit_rejects_a_data_keyword_even_with_no_listener(settings_factory: Any) -> None:
    """The guard sits above the early return, so a silent run cannot hide the bug."""
    orchestrator = make_orchestrator(settings_factory())

    assert orchestrator._on_event is None
    with pytest.raises(TypeError) as excinfo:
        orchestrator._emit(EventType.ERROR, data={"x": 1})

    assert "splat" in str(excinfo.value)


def test_emit_rejects_an_emptily_splatted_data_payload(settings_factory: Any) -> None:
    """``**{"data": ...}`` binds the same keyword, so it is caught too."""
    orchestrator = make_orchestrator(settings_factory())

    with pytest.raises(TypeError):
        orchestrator._emit(EventType.ERROR, **{"data": {"x": 1}})


def test_a_normal_emit_keeps_the_payload_flat_beside_its_siblings(
    settings_factory: Any,
) -> None:
    events: list[FurrowEvent] = []
    orchestrator = make_orchestrator(settings_factory(), on_event=events.append)

    orchestrator._emit(EventType.ERROR, message="m", phase="test")

    assert len(events) == 1
    assert events[0].type is EventType.ERROR
    assert events[0].message == "m"
    assert events[0].data == {"phase": "test"}


def test_nested_values_under_other_keys_are_still_allowed(settings_factory: Any) -> None:
    """Only the literal ``data`` key is banned; a dict under any other key is fine."""
    events: list[FurrowEvent] = []
    orchestrator = make_orchestrator(settings_factory(), on_event=events.append)

    orchestrator._emit(EventType.ERROR, message="m", detail={"nested": True})

    assert events[0].data == {"detail": {"nested": True}}


def test_emit_without_a_listener_is_a_no_op(settings_factory: Any) -> None:
    orchestrator = make_orchestrator(settings_factory())

    assert orchestrator._emit(EventType.ERROR, message="m", phase="test") is None


async def test_the_data_guard_fires_on_the_real_run_path(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """The guard is on the production call path, not only in isolation."""
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events: list[FurrowEvent] = []
    orchestrator = make_orchestrator(settings, on_event=events.append)
    real_emit = orchestrator._emit
    calls = 0

    def guarded_emit(event_type: EventType, *args: Any, **data: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            with pytest.raises(TypeError):
                real_emit(event_type, data={"nested": True})
        real_emit(event_type, *args, **data)

    orchestrator._emit = guarded_emit

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert calls > 1
    assert result.completed is True
    assert [event for event in events if "data" in event.data] == []


async def test_event_types_that_carry_payload_are_flat_in_the_same_way(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """Two payload-bearing types agree on shape: keys sit at the top level."""
    events = await _every_event_type_in_one_run(monkeypatch, settings_factory(max_cycles=1))
    payloads = {
        event.type: event.data for event in events if event.type is not EventType.RUN_START
    }

    assert EventType.TEST_RESULT in payloads
    assert EventType.RUN_END in payloads
    verdict, run_end = payloads[EventType.TEST_RESULT], payloads[EventType.RUN_END]
    assert verdict["passed"] is True
    assert run_end["completed"] is False
    assert not {"data"} & set(verdict)
    assert not {"data"} & set(run_end)
    assert all(not isinstance(value, dict) for value in (*verdict.values(), *run_end.values()))


# --------------------------------------------------------------------------- #
# client lifecycle
# --------------------------------------------------------------------------- #


async def test_client_release_is_called_after_a_normal_run(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    orchestrator = make_orchestrator(settings)

    await orchestrator.run()

    assert isinstance(orchestrator.client, FakeLLMClient)
    assert orchestrator.client.released == 1


class _FatalRun(BaseException):
    """A ``BaseException`` no ``except Exception`` clause will catch.

    Using it keeps the "run raises" case out of pytest's ``KeyboardInterrupt``
    handling while still escaping :meth:`Orchestrator.run` and therefore
    exercising the ``finally`` block.
    """


async def test_client_release_is_called_when_the_run_raises(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    """``release`` must be in a ``finally``, not on the happy path only."""
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    orchestrator = make_orchestrator(settings)

    async def exploding_cycle() -> None:
        raise _FatalRun("aborted hard")

    monkeypatch.setattr(orchestrator, "_cycle", exploding_cycle)

    with pytest.raises(_FatalRun):
        await orchestrator.run()

    assert orchestrator.client.released == 1


async def test_client_release_is_called_on_cancellation(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events: list[FurrowEvent] = []
    orchestrator = make_orchestrator(settings, on_event=events.append)

    async def slow_cycle() -> None:
        await asyncio.sleep(30)

    monkeypatch.setattr(orchestrator, "_cycle", slow_cycle)
    task = asyncio.create_task(orchestrator.run())
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert orchestrator.client.released == 1
    cancelled = [event for event in events if event.data.get("cancelled") is True]
    assert cancelled, [event.data for event in events]


async def test_a_failing_release_does_not_mask_the_run_result(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)
    events: list[FurrowEvent] = []
    orchestrator = make_orchestrator(settings, on_event=events.append)

    def broken_release() -> None:
        raise RuntimeError("release failed")

    monkeypatch.setattr(orchestrator.client, "release", broken_release)

    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert result.completed is True
    assert any(event.data.get("phase") == "release" for event in events)


async def test_a_client_without_release_is_tolerated(
    monkeypatch: pytest.MonkeyPatch, settings_factory: Any
) -> None:
    class _NoRelease:
        """A client that simply has no ``release`` attribute."""

        settings: Settings

        def __init__(self, settings: Settings) -> None:
            self.settings = settings

    planner = ScriptedPlanner([_plan(("1", []))])
    tester = ScriptedTester([_passing()])
    install_agents(monkeypatch, planner, tester)
    settings = settings_factory(max_cycles=1)

    orchestrator = Orchestrator(
        goal="make it work", client=_NoRelease(settings), settings=settings
    )
    result = await asyncio.wait_for(orchestrator.run(), timeout=30)

    assert result.completed is True
