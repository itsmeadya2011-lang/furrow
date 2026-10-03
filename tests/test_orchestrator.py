"""The loop itself: how many cycles it runs, in what order, and how it reports its ending.

The agents are constructed *inside* ``Orchestrator._cycle``/``_run_task``, so the module
attributes ``furrow.core.orchestrator.{PlannerAgent,WorkerAgent,TesterAgent}`` are the only
clean seam. Every test here replaces those three with scripted doubles and asserts on
observable behaviour (task ordering, task statuses, cycle counts, the returned
:class:`~furrow.core.orchestrator.RunOutcome`) rather than on printed output.

No test waits on a real timeout: ``run()`` is always wrapped in :func:`run_bounded`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import (
    HALT_GOAL_COMPLETE,
    HALT_NO_PROGRESS,
    HALT_PLANNER_FAILED,
    MAX_NO_PROGRESS_CYCLES,
    Orchestrator,
)

from .conftest import FakeLLMClient

# --------------------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------------------


class TaskRecorder:
    """Records the start/finish order and peak concurrency of the fake workers."""

    def __init__(self, *, delay: float = 0.0, failing: set[str] | None = None) -> None:
        self.delay = delay
        self.failing = failing or set()
        #: ``"start:<id>"`` / ``"finish:<id>"`` entries, in the order they happened.
        self.order: list[str] = []
        self.peak_concurrency = 0
        self._live = 0

    async def run_task(self, task: TaskModel) -> str:
        self._live += 1
        self.peak_concurrency = max(self.peak_concurrency, self._live)
        self.order.append(f"start:{task.id}")
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if task.id in self.failing:
                raise RuntimeError(f"worker exploded on {task.id}")
            return f"wrote {task.id}"
        finally:
            self._live -= 1
            self.order.append(f"finish:{task.id}")


class ScriptedPlanner:
    """A ``PlannerAgent`` stand-in driven by one scripted reply per cycle.

    An exhausted script raises :class:`AssertionError`, which the orchestrator treats as a
    planner failure -- so an over-running loop shows up as a halt reason, never as a pass.
    """

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls = 0
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        index = self.calls
        self.calls += 1
        self.goals.append(goal)
        if index >= len(self._script):
            raise AssertionError(
                f"planner called {self.calls} time(s), only {len(self._script)} replies scripted"
            )
        reply = self._script[index]
        if isinstance(reply, BaseException):
            raise reply
        return reply


class ScriptedTester:
    """A ``TesterAgent`` stand-in returning one scripted :class:`TestResult` per cycle."""

    def __init__(self, script: list[Any]) -> None:
        self._script = list(script)
        self.calls = 0
        self.contexts: list[str] = []
        self.task_lists: list[list[TaskModel]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        index = self.calls
        self.calls += 1
        self.contexts.append(goal)
        self.task_lists.append(list(tasks))
        if index >= len(self._script):
            raise AssertionError(
                f"tester called {self.calls} time(s), only {len(self._script)} results scripted"
            )
        reply = self._script[index]
        if isinstance(reply, BaseException):
            raise reply
        return reply


class EndlessPlanner:
    """A ``PlannerAgent`` stand-in with an inexhaustible plan source.

    :class:`ScriptedPlanner` raises once its script runs out, which ends a run through the
    planner-outage path and would *hide* whether the anti-spin guard works. This double keeps
    planning forever, so the guard is the only thing that can stop such a run -- exactly the
    situation the guard exists for.
    """

    def __init__(self, plan_factory: Callable[[], Plan]) -> None:
        self._plan_factory = plan_factory
        self.calls = 0
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        self.goals.append(goal)
        return self._plan_factory()


class EndlessTester:
    """A ``TesterAgent`` stand-in that answers every cycle with the same result."""

    def __init__(self, result: TestResult) -> None:
        self._result = result
        self.calls = 0
        self.contexts: list[str] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.calls += 1
        self.contexts.append(goal)
        return self._result


def install_planner(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> None:
    """Replace the orchestrator's planner with a scripted one."""

    def _planner(client: Any = None, settings: Any = None) -> ScriptedPlanner:
        return ScriptedPlanner(script)

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", _planner)


def install_worker(monkeypatch: pytest.MonkeyPatch, recorder: TaskRecorder) -> None:
    """Replace the orchestrator's worker with one that records instead of writing files."""

    class _Worker:
        def __init__(self, task: TaskModel, client: Any = None, settings: Any = None) -> None:
            self.task = task

        async def run(self) -> str:
            return await recorder.run_task(self.task)

    monkeypatch.setattr(orchestrator_module, "WorkerAgent", _Worker)


def install_tester(monkeypatch: pytest.MonkeyPatch, script: list[Any]) -> ScriptedTester:
    """Replace the orchestrator's tester with a scripted one.

    ``_test`` builds a fresh ``TesterAgent`` on every cycle, so the double must be
    *shared*: one :class:`ScriptedTester`, handing out the next scripted result each
    time. Handing out a new double per cycle would reset its cursor and replay
    ``script[0]`` forever, so a test could never observe what happens on cycle two.

    Returns:
        The shared double, so a test can assert on the contexts it was handed.
    """

    tester = ScriptedTester(script)

    def _tester(client: Any = None, settings: Any = None) -> ScriptedTester:
        return tester

    monkeypatch.setattr(orchestrator_module, "TesterAgent", _tester)
    return tester


def install_endless_planner(
    monkeypatch: pytest.MonkeyPatch, plan_factory: Callable[[], Plan]
) -> EndlessPlanner:
    """Replace the planner with one that can plan forever, drawing from ``plan_factory``.

    Args:
        monkeypatch: The pytest fixture used to swap the agent class.
        plan_factory: Called once per cycle to produce that cycle's plan.

    Returns:
        The shared double, so a test can assert on how often it was asked for a plan.
    """

    planner = EndlessPlanner(plan_factory)

    def _planner(client: Any = None, settings: Any = None) -> EndlessPlanner:
        return planner

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", _planner)
    return planner


def install_endless_tester(
    monkeypatch: pytest.MonkeyPatch, result: TestResult
) -> EndlessTester:
    """Replace the tester with one that always answers ``result``."""

    tester = EndlessTester(result)

    def _tester(client: Any = None, settings: Any = None) -> EndlessTester:
        return tester

    monkeypatch.setattr(orchestrator_module, "TesterAgent", _tester)
    return tester


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def one_task_plan(
    task_id: str = "t", description: str = "do the work", dependencies: list[str] | None = None
) -> Plan:
    """A plan with a single runnable task."""
    return Plan(
        tasks=[TaskModel(id=task_id, description=description, dependencies=dependencies or [])],
        rationale="because",
    )


def empty_plan() -> Plan:
    """A plan with no tasks: the planner's "goal already done" answer."""
    return Plan(tasks=[], rationale="nothing left to do")


def alternating_plans(*plans: Plan) -> Callable[[], Plan]:
    """A plan factory that hands out ``plans`` in order and then starts over, forever."""

    index = 0

    def factory() -> Plan:
        nonlocal index
        plan = plans[index % len(plans)]
        index += 1
        return plan

    return factory


def passing_result(summary: str = "12 passed") -> TestResult:
    return TestResult(passed=True, summary=summary, failures=[])


def failing_result(*failures: str) -> TestResult:
    listed = list(failures) or ["tests/test_api.py::test_boom - AssertionError"]
    return TestResult(passed=False, summary=f"{len(listed)} failed", failures=listed)


def make_orchestrator(
    goal: str = "ship the feature", *, on_event: Any = None, **settings_kwargs: Any
) -> Orchestrator:
    """An orchestrator with deterministic settings and no provider traffic.

    ``on_event`` is a constructor argument of :class:`Orchestrator`, not a
    :class:`~furrow.config.Settings` field, so it is split off here: ``Settings``
    forbids extra inputs and would reject it outright.
    """
    settings_kwargs.setdefault("max_cycles", 1)
    settings = Settings(**settings_kwargs)
    client = FakeLLMClient(settings=settings)
    return Orchestrator(goal=goal, client=client, settings=settings, on_event=on_event)


async def run_bounded(orchestrator: Orchestrator, timeout: float = 10.0):
    """Run the loop under a hard timeout: a hang must fail the test, not the suite."""
    return await asyncio.wait_for(orchestrator.run(), timeout=timeout)


# --------------------------------------------------------------------------------------
# 1. the loop must run more than one cycle
# --------------------------------------------------------------------------------------


async def test_unlimited_run_executes_several_cycles_then_stops_on_the_anti_spin_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unlimited run with nothing to do must spin a bounded number of times, then halt."""
    no_progress = MAX_NO_PROGRESS_CYCLES
    recorder = TaskRecorder(failing={"t"})
    install_planner(monkeypatch, [one_task_plan() for _ in range(no_progress)])
    install_worker(monkeypatch, recorder)
    install_tester(monkeypatch, [passing_result() for _ in range(no_progress)])

    orchestrator = make_orchestrator("make the suite green", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == no_progress
    assert outcome.cycles > 1
    assert orchestrator.cycles == outcome.cycles
    assert "No new work in" in outcome.stop_reason
    assert orchestrator.stop_reason == outcome.stop_reason
    # Every cycle planned fresh work, so nothing was actually finished.
    assert orchestrator.completed_total == 0
    assert len(orchestrator.tasks) == no_progress
    assert recorder.order.count("start:t") == no_progress
    assert orchestrator.planner.calls == no_progress


async def test_unlimited_run_halts_when_the_planner_only_repeats_completed_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The adversarial case the guard exists for: productive-looking plans, forever.

    Every cycle completes a task and every test passes, so no individual cycle looks
    unhealthy and the anti-spin guard is the only thing that can end the run. A guard
    counting raw completions instead of *new* completions would let this spin forever, which
    is why the run is wrapped in :func:`run_bounded`: without the guard it hangs rather than
    failing quickly.
    """
    recorder = TaskRecorder()
    planner = install_endless_planner(monkeypatch, one_task_plan)
    install_worker(monkeypatch, recorder)
    tester = install_endless_tester(monkeypatch, passing_result())

    orchestrator = make_orchestrator("planner never runs dry", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    # Cycle one does genuinely new work; every cycle after it only re-does that same work,
    # so the run ends exactly MAX_NO_PROGRESS_CYCLES cycles later -- not before, not after.
    assert outcome.cycles == MAX_NO_PROGRESS_CYCLES + 1
    assert "No new work in" in outcome.stop_reason
    assert orchestrator._halt().kind == HALT_NO_PROGRESS
    # Real work was completed on every cycle, so the stall is a partial success.
    assert outcome.succeeded is True
    assert outcome.completed == outcome.cycles
    assert planner.calls == outcome.cycles
    assert tester.calls == outcome.cycles
    assert recorder.order.count("start:t") == outcome.cycles


async def test_repeating_two_alternating_descriptions_still_trips_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alternating between two tasks is one stall, not new work every other cycle.

    Both descriptions are completed by the second cycle, so from then on the planner only
    re-issues the same two units of work. Counting cycles that completed *something* would
    reset the guard on every one of them.
    """
    recorder = TaskRecorder()
    planner = install_endless_planner(
        monkeypatch,
        alternating_plans(one_task_plan("a", "work a"), one_task_plan("b", "work b")),
    )
    install_worker(monkeypatch, recorder)
    install_endless_tester(monkeypatch, passing_result())

    orchestrator = make_orchestrator("two alternating tasks", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == MAX_NO_PROGRESS_CYCLES + 2
    assert "No new work in" in outcome.stop_reason
    assert orchestrator._halt().kind == HALT_NO_PROGRESS
    assert outcome.succeeded is True
    # Both cycles of new work were really done, then the repeats kept completing a task.
    assert {task.description for task in orchestrator.tasks} == {"work a", "work b"}
    assert outcome.completed == outcome.cycles
    assert planner.calls == outcome.cycles
    assert recorder.order.count("start:a") + recorder.order.count("start:b") == outcome.cycles


async def test_repeating_the_same_stall_with_a_failing_worker_is_not_a_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The paired negative control: the same stall shape, but nothing was ever achieved.

    The planner keeps offering work and the tests stay green, so the guard fires exactly as
    above -- yet with every worker raising, no task was ever completed and the run must be
    reported as a failure rather than as a partial win.
    """
    recorder = TaskRecorder(failing={"t"})
    planner = install_endless_planner(monkeypatch, one_task_plan)
    install_worker(monkeypatch, recorder)
    install_endless_tester(monkeypatch, passing_result())

    orchestrator = make_orchestrator("stalled on failing work", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    # Nothing completed, so not even the first cycle counted as progress.
    assert outcome.cycles == MAX_NO_PROGRESS_CYCLES
    assert "No new work in" in outcome.stop_reason
    assert orchestrator._halt().kind == HALT_NO_PROGRESS
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is True
    assert outcome.completed == 0
    assert outcome.succeeded is False
    assert planner.calls == outcome.cycles
    assert recorder.order.count("start:t") == outcome.cycles


async def test_pending_work_alone_does_not_halt_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Three cycles of new work must not trip the guard; only the empty plan may end the run.

    ``_halt()`` is sampled from inside the ``plan`` event -- after the plan exists but before
    any of it has been executed -- so each snapshot answers the real question: "with this
    work still pending, would the loop stop?" While work is pending the answer must be no;
    the only cycle where stopping is correct is the one whose plan is empty.

    This is the direct replacement for the old ``_get_tasks``/``_is_done`` guard. That test's
    docstring claimed it guarded the single-cycle halt, but pending work was never the stop
    condition: the halt bug lived in the stop condition reading an empty task list, and
    ``_get_tasks`` returning ``[]`` only made it visible.
    """
    productive = [
        one_task_plan("t1", "write the parser"),
        one_task_plan("t2", "write the renderer"),
        one_task_plan("t3", "write the docs"),
    ]
    install_planner(monkeypatch, [*productive, empty_plan()])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [passing_result() for _ in range(len(productive) + 1)])

    orchestrator: Orchestrator | None = None
    sampled: list[tuple[int, Any]] = []

    def on_event(name: str, payload: dict[str, Any]) -> None:
        if name != "plan":
            return
        assert orchestrator is not None
        sampled.append((payload["cycle"], orchestrator._halt()))

    orchestrator = make_orchestrator(
        "three rounds of new work", max_cycles=0, on_event=on_event
    )
    outcome = await run_bounded(orchestrator)

    assert [cycle for cycle, _ in sampled] == [1, 2, 3, 4]
    # Cycles one to three each had real, distinct work outstanding and still did not stop:
    # pending work is never on its own a reason to halt.
    assert [halt for _, halt in sampled[:3]] == [None, None, None]
    # Cycle four planned nothing while the tests were still green, so stopping there is
    # correct -- and it must be the goal-complete path, not the anti-spin guard, whose streak
    # was reset by cycle three.
    final_halt = sampled[3][1]
    assert final_halt is not None
    assert final_halt.kind == HALT_GOAL_COMPLETE
    assert final_halt.reason == "Goal complete. Halting."
    assert len({task.description for task in orchestrator.tasks}) == len(productive)
    # And the run ended on the goal, not on the guard.
    assert outcome.cycles == len(productive) + 1
    assert outcome.stop_reason == "Goal complete. Halting."
    assert orchestrator._halt().kind == HALT_GOAL_COMPLETE
    assert outcome.succeeded is True
    assert outcome.completed == len(productive)


async def test_max_cycles_stops_the_run_after_exactly_that_many_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``max_cycles`` used to be ignored entirely."""
    recorder = TaskRecorder()
    install_planner(monkeypatch, [one_task_plan() for _ in range(6)])
    install_worker(monkeypatch, recorder)
    install_tester(monkeypatch, [passing_result() for _ in range(6)])

    orchestrator = make_orchestrator("bounded run", max_cycles=2)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == 2
    assert outcome.stop_reason == "Reached max_cycles (2). Halting."
    assert orchestrator.planner.calls == 2
    assert recorder.order == ["start:t", "finish:t"] * 2


# --------------------------------------------------------------------------------------
# 2-3. dependencies
# --------------------------------------------------------------------------------------


async def test_dependent_task_waits_for_its_dependency_to_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorded ordering, not an assumption: ``second`` starts only after ``first`` finished."""
    plan = Plan(
        tasks=[
            TaskModel(id="first", description="create the schema"),
            TaskModel(id="second", description="read the schema", dependencies=["first"]),
        ],
        rationale="serial",
    )
    recorder = TaskRecorder(delay=0.01)
    install_planner(monkeypatch, [plan])
    install_worker(monkeypatch, recorder)
    install_tester(monkeypatch, [passing_result()])

    orchestrator = make_orchestrator("schema then consumer")
    outcome = await run_bounded(orchestrator)

    assert recorder.order == ["start:first", "finish:first", "start:second", "finish:second"]
    assert [task.status for task in plan.tasks] == ["completed", "completed"]
    assert orchestrator.completed_total == 2
    assert outcome.succeeded is True


async def test_independent_tasks_do_run_concurrently(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control for the ordering test: with no edge between them, both are in flight."""
    plan = Plan(
        tasks=[TaskModel(id="a", description="a"), TaskModel(id="b", description="b")],
        rationale="parallel",
    )
    recorder = TaskRecorder(delay=0.01)
    install_planner(monkeypatch, [plan])
    install_worker(monkeypatch, recorder)
    install_tester(monkeypatch, [passing_result()])

    await run_bounded(make_orchestrator("two independent tasks"))

    assert recorder.peak_concurrency == 2
    assert recorder.order == ["start:a", "start:b", "finish:a", "finish:b"]


async def test_task_with_a_failed_dependency_is_skipped_never_completed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="first", description="flaky build"),
            TaskModel(id="second", description="needs first", dependencies=["first"]),
        ],
        rationale="serial",
    )
    recorder = TaskRecorder(failing={"first"})
    install_planner(monkeypatch, [plan])
    install_worker(monkeypatch, recorder)
    install_tester(monkeypatch, [passing_result()])

    orchestrator = make_orchestrator("serial work")
    await run_bounded(orchestrator)

    first, second = plan.tasks
    assert first.status == "failed"
    assert second.status == "skipped"
    assert "first" in second.result
    assert recorder.order == ["start:first", "finish:first"]
    assert orchestrator.completed_total == 0


# --------------------------------------------------------------------------------------
# 5. parallelism
# --------------------------------------------------------------------------------------


async def test_max_parallel_tasks_bounds_observed_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Six tasks with a limit of two must never put more than two in flight at once."""
    plan = Plan(
        tasks=[TaskModel(id=f"t{index}", description=f"task {index}") for index in range(6)],
        rationale="wide",
    )
    recorder = TaskRecorder(delay=0.02)
    install_planner(monkeypatch, [plan])
    install_worker(monkeypatch, recorder)
    install_tester(monkeypatch, [passing_result()])

    await run_bounded(make_orchestrator("wide plan", max_parallel_tasks=2))

    assert recorder.peak_concurrency == 2
    assert len(recorder.order) == 12
    assert all(task.status == "completed" for task in plan.tasks)


# --------------------------------------------------------------------------------------
# 6. the goal survives a failing test run
# --------------------------------------------------------------------------------------


async def test_original_goal_survives_a_failing_test_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """The working focus is refined; the original goal must not be overwritten by it."""
    goal = "Add a /health endpoint"
    failure = "tests/test_api.py::test_health - AssertionError: 404 != 200"
    install_planner(monkeypatch, [one_task_plan(), one_task_plan()])
    install_worker(monkeypatch, TaskRecorder())
    tester = install_tester(monkeypatch, [failing_result(failure), failing_result(failure)])

    orchestrator = make_orchestrator(goal, max_cycles=2)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == 2
    # ``goal`` is immutable for the whole run; only the working focus carries the failures.
    assert orchestrator.goal == goal
    assert orchestrator.focus != goal
    assert goal in orchestrator.focus
    assert failure in orchestrator.focus
    assert "Fix failing tests" in orchestrator.focus
    # The failures are carried into the *next* cycle rather than lost: the planner is handed
    # the refined focus, and the tester is told both the original objective and that focus,
    # so a model that lost the thread still sees the real goal.
    assert orchestrator.planner.goals[0] == goal
    assert orchestrator.planner.goals[1] == orchestrator.focus
    assert failure in orchestrator.planner.goals[1]
    assert tester.contexts[1].startswith(f"Original goal:\n{goal}")
    assert f"Current focus:\n{orchestrator.focus}" in tester.contexts[1]
    assert failure in tester.contexts[1]
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is False


async def test_goal_is_preserved_across_failing_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    goal = "Delete the legacy importer"
    install_planner(monkeypatch, [one_task_plan() for _ in range(2)])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [failing_result("first failure"), failing_result("last failure")])

    orchestrator = make_orchestrator(goal, max_cycles=2)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == 2
    assert orchestrator.goal == goal
    assert orchestrator.goal in orchestrator.focus
    assert "last failure" in orchestrator.focus


# --------------------------------------------------------------------------------------
# 7. malformed dependency graphs
# --------------------------------------------------------------------------------------


async def test_unknown_dependency_id_does_not_block_the_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = one_task_plan(dependencies=["ghost"])
    recorder = TaskRecorder()
    install_planner(monkeypatch, [plan])
    install_worker(monkeypatch, recorder)
    install_tester(monkeypatch, [passing_result()])

    await run_bounded(make_orchestrator("dangling edge"))

    assert plan.tasks[0].status == "completed"
    assert recorder.order == ["start:t", "finish:t"]


async def test_dependency_cycle_is_survivable_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cycle used to be able to hang the level scheduler; it must instead be dropped."""
    no_progress = MAX_NO_PROGRESS_CYCLES
    plan = Plan(
        tasks=[
            TaskModel(id="a", description="a", dependencies=["b"]),
            TaskModel(id="b", description="b", dependencies=["a"]),
        ],
        rationale="cyclic",
    )
    install_planner(monkeypatch, [plan for _ in range(no_progress)])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [passing_result() for _ in range(no_progress)])

    orchestrator = make_orchestrator("cyclic plan", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert [task.status for task in plan.tasks] == ["skipped", "skipped"]
    assert orchestrator.completed_total == 0
    # It stopped because of the anti-spin guard, not because it hung.
    assert outcome.cycles == no_progress
    assert "No new work in" in outcome.stop_reason


def test_dependency_levels_drops_self_and_unknown_edges() -> None:
    """One normalised graph, so the wave layout and the readiness check cannot disagree.

    :meth:`Orchestrator._normalised_graph` is the single source of truth for a plan's
    edges: a self reference and a dangling id are both dropped, so neither can hold a task
    back, and a repeated edge is reported once. This is the guard for the bug where
    ``_dependency_levels`` *scheduled* a self-dependent task and ``_blocked_by`` then
    *skipped* it on the strength of its own ``pending`` status -- one plan, two verdicts,
    and the task vanished without running.
    """
    plan = Plan(
        tasks=[
            TaskModel(id="a", description="a"),
            TaskModel(id="b", description="b", dependencies=["a", "b", "ghost", "a"]),
            TaskModel(id="c", description="c", dependencies=["ghost"]),
        ],
        rationale="malformed",
    )
    orchestrator = make_orchestrator()

    graph = orchestrator._normalised_graph(plan.tasks)
    assert graph == {"a": [], "b": ["a"], "c": []}

    levels = orchestrator._dependency_levels(plan.tasks, graph)
    assert [[task.id for task in level] for level in levels] == [["a", "c"], ["b"]]
    # Omitting the graph must reach the same layout: both consumers normalise the same way.
    assert orchestrator._dependency_levels(plan.tasks) == levels

    # The readiness check reads the same edges, so it agrees with the layout: ``b`` waits on
    # its one real dependency ``a`` and nothing else -- never on itself, never on the ghost.
    by_id = {task.id: task for task in plan.tasks}
    assert orchestrator._blocked_by(plan.tasks[1], by_id, graph) == "a (pending)"
    assert orchestrator._blocked_by(plan.tasks[2], by_id, graph) is None
    # The ``deps=None`` fallback must filter the same unusable edges, not just the graph path.
    assert orchestrator._blocked_by(plan.tasks[2], by_id) is None
    assert orchestrator._blocked_by(plan.tasks[1], by_id) == "a (pending)"
    # And with ``a`` completed, ``b`` is free: the dropped self edge cannot hold it back.
    by_id["a"].status = "completed"
    assert orchestrator._blocked_by(plan.tasks[1], by_id, graph) is None


async def test_self_and_unknown_edges_leave_the_run_survivable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A task whose edges were all dropped still runs, and the run still halts normally."""
    plan = Plan(
        tasks=[
            TaskModel(id="t", description="self edge plus a dangling one", dependencies=["t", "ghost"]),
            TaskModel(id="u", description="dangling edge only", dependencies=["ghost"]),
        ],
        rationale="malformed",
    )
    install_planner(monkeypatch, [plan])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [passing_result()])

    orchestrator = make_orchestrator("self edge")
    outcome = await run_bounded(orchestrator)

    # Both tasks reached a terminal status. ``completed`` can only be set once the worker
    # actually ran, so the self-dependent task is guarded here without pinning down *when*
    # it ran -- the level scheduler may legitimately reorder the two.
    assert [task.status for task in plan.tasks] == ["completed", "completed"]
    assert all(task.result for task in plan.tasks)
    assert orchestrator.completed_total == 2
    # And the run ended on its own terms instead of spinning on the unusable edges.
    assert outcome.cycles == 1
    assert outcome.stop_reason == "Reached max_cycles (1). Halting."
    assert outcome.succeeded is True


# --------------------------------------------------------------------------------------
# 8. the progress hook
# --------------------------------------------------------------------------------------


async def test_on_event_receives_the_whole_cycle_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    payloads: dict[str, list[dict[str, Any]]] = {}
    plan = one_task_plan()

    def on_event(name: str, payload: dict[str, Any]) -> None:
        events.append(name)
        payloads.setdefault(name, []).append(payload)

    install_planner(monkeypatch, [plan])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [passing_result()])

    orchestrator = make_orchestrator("observable run", on_event=on_event)
    outcome = await run_bounded(orchestrator)

    assert events == ["cycle_start", "plan", "task", "test", "cycle_end"]
    assert payloads["cycle_start"][0] == {"cycle": 1}
    assert payloads["plan"][0]["task_count"] == 1
    assert payloads["task"][0]["id"] == "t"
    assert payloads["task"][0]["status"] == "completed"
    assert payloads["test"][0]["passed"] is True
    assert payloads["cycle_end"][0] == {
        "cycle": 1,
        "completed": 1,
        "completed_total": 1,
        "tests_passed": True,
    }
    assert outcome.completed == 1


async def test_raising_on_event_hook_does_not_break_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = one_task_plan()
    seen: list[str] = []

    def on_event(name: str, payload: dict[str, Any]) -> None:
        seen.append(name)
        raise RuntimeError(f"hook exploded during {name}")

    install_planner(monkeypatch, [plan])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [passing_result()])

    orchestrator = make_orchestrator("hostile hook", on_event=on_event)
    outcome = await run_bounded(orchestrator)

    assert seen == ["cycle_start", "plan", "task", "test", "cycle_end"]
    assert outcome.cycles == 1
    assert orchestrator.completed_total == 1
    assert plan.tasks[0].status == "completed"


async def test_hook_failure_does_not_hide_a_planner_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures = orchestrator_module.MAX_PLANNER_FAILURES
    install_planner(monkeypatch, [RuntimeError("planner is down") for _ in range(failures)])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [])

    def on_event(name: str, payload: dict[str, Any]) -> None:
        raise RuntimeError("every event explodes")

    orchestrator = make_orchestrator("doomed run", max_cycles=0, on_event=on_event)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == failures
    assert "Planner failed" in outcome.stop_reason
    assert outcome.succeeded is False


# --------------------------------------------------------------------------------------
# 9. planner exception safety
# --------------------------------------------------------------------------------------


async def test_transient_planner_failure_is_retried_on_a_later_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = one_task_plan()
    recorder = TaskRecorder()
    install_planner(monkeypatch, [ValueError("unusable plan"), plan])
    install_worker(monkeypatch, recorder)
    install_tester(monkeypatch, [passing_result()])

    orchestrator = make_orchestrator("flaky planner", max_cycles=2)
    outcome = await run_bounded(orchestrator)

    assert orchestrator.planner.calls == 2
    assert outcome.cycles == 2
    assert [task.status for task in plan.tasks] == ["completed"]
    assert orchestrator.completed_total == 1


async def test_a_permanently_failing_planner_halts_the_run_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planner outage must stop the run, not spin it forever."""
    failures = orchestrator_module.MAX_PLANNER_FAILURES
    install_planner(monkeypatch, [RuntimeError("planner is down") for _ in range(failures)])
    install_worker(monkeypatch, TaskRecorder())
    # An empty tester script is a tripwire: the loop must never get as far as testing.
    install_tester(monkeypatch, [])

    orchestrator = make_orchestrator("doomed run", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == failures
    assert orchestrator.planner.calls == failures
    assert orchestrator.completed_total == 0
    assert orchestrator.outcome is outcome


# --------------------------------------------------------------------------------------
# 10. RunOutcome
# --------------------------------------------------------------------------------------


async def test_outcome_reports_success_when_the_goal_is_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_planner(monkeypatch, [empty_plan()])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [passing_result()])

    orchestrator = make_orchestrator("already done", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert outcome.stop_reason == "Goal complete. Halting."
    # ``_stop_reason()`` is the *message* view of ``_halt()``; the ``HALT_*`` constants are
    # the ``kind`` half of the same decision, so only ``_halt()`` can answer that question.
    assert orchestrator._halt().kind == HALT_GOAL_COMPLETE
    assert orchestrator._stop_reason() == outcome.stop_reason == "Goal complete. Halting."
    assert outcome.succeeded is True
    assert orchestrator.succeeded is True
    assert outcome == orchestrator.outcome


async def test_outcome_reports_failure_for_a_planner_outage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failures = orchestrator_module.MAX_PLANNER_FAILURES
    install_planner(monkeypatch, [RuntimeError("planner is down") for _ in range(failures)])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [])

    orchestrator = make_orchestrator("doomed run", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert f"Planner failed {failures} cycles in a row" in outcome.stop_reason
    # Same split as above: the constant is the halt kind, the message is what the CLI shows.
    assert orchestrator._halt().kind == HALT_PLANNER_FAILED
    assert orchestrator._stop_reason() == outcome.stop_reason
    assert outcome.succeeded is False
    assert orchestrator.succeeded is False


async def test_outcome_reports_partial_success_after_the_anti_spin_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running out of ideas after doing real work with green tests is not a failure."""
    no_progress = MAX_NO_PROGRESS_CYCLES
    install_planner(monkeypatch, [one_task_plan("good"), *[one_task_plan("bad")] * no_progress])
    install_worker(monkeypatch, TaskRecorder(failing={"bad"}))
    install_tester(monkeypatch, [passing_result() for _ in range(no_progress + 1)])

    orchestrator = make_orchestrator("partial win", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == no_progress + 1
    assert "No new work in" in outcome.stop_reason
    assert orchestrator.completed_total == 1
    assert outcome.completed == 1
    assert outcome.succeeded is True
    assert orchestrator.succeeded is True


async def test_outcome_reports_failure_when_the_anti_spin_guard_found_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    no_progress = MAX_NO_PROGRESS_CYCLES
    install_planner(monkeypatch, [empty_plan() for _ in range(no_progress)])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [failing_result() for _ in range(no_progress)])

    orchestrator = make_orchestrator("no progress at all", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == no_progress
    assert outcome.completed == 0
    assert "No new work in" in outcome.stop_reason
    assert outcome.succeeded is False


async def test_outcome_is_none_before_the_run_finishes(monkeypatch: pytest.MonkeyPatch) -> None:
    install_planner(monkeypatch, [one_task_plan()])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [passing_result()])

    orchestrator = make_orchestrator("not started")
    assert orchestrator.outcome is None
    assert orchestrator.succeeded is False

    await run_bounded(orchestrator)

    assert orchestrator.outcome is not None
    assert orchestrator.succeeded is True


async def test_a_broken_tester_run_does_not_kill_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_planner(monkeypatch, [one_task_plan(), one_task_plan()])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [RuntimeError("tester exploded"), passing_result()])

    orchestrator = make_orchestrator("survives a broken tester", max_cycles=2)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == 2
    assert orchestrator.completed_total == 2
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is True


async def test_test_result_defaults_to_passing_while_the_tester_keeps_breaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tester that always raises leaves the run with no verdict, which is not a success."""
    # An empty plan cannot complete the goal on its own -- the run has no passing verdict,
    # so nothing but the anti-spin guard can stop it, and that fires on the cycle where the
    # no-progress streak reaches MAX_NO_PROGRESS_CYCLES. Four cycles would mean the guard
    # over-counts.
    cycles = MAX_NO_PROGRESS_CYCLES
    install_planner(monkeypatch, [empty_plan() for _ in range(cycles)])
    install_worker(monkeypatch, TaskRecorder())
    install_tester(monkeypatch, [RuntimeError("tester exploded") for _ in range(cycles)])

    orchestrator = make_orchestrator("broken tester only", max_cycles=0)
    outcome = await run_bounded(orchestrator)

    assert outcome.cycles == cycles
    assert outcome.cycles > 1
    assert orchestrator.last_test_result is None
    assert "No new work in" in outcome.stop_reason
    assert outcome.succeeded is False