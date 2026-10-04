"""Behavioural tests for :mod:`furrow.core.orchestrator` — the Furrow loop.

The orchestrator *is* the product. Furrow is a bounded parallel development
loop, so a loop that runs exactly one cycle and prints "Goal complete. Halting."
is not a cosmetic bug: the defining feature does nothing at all. The heaviest
test in this module (``test_loop_runs_every_cycle_in_the_budget``) therefore
pins the number of cycles to the cycle budget, and
``test_is_done_is_false_while_work_remains`` pins the reason that used to be
wrong: ``_get_tasks()`` returned ``[]`` before a plan existed, ``completed >=
len(tasks)`` evaluated ``0 >= 0``, and ``_is_done()`` answered True on the very
first call.

Everything here runs offline. :func:`install_fakes` replaces the three agents
the loop drives (``PlannerAgent``, ``WorkerAgent``, ``TesterAgent``) with
recording stubs, so no real agent is ever constructed, no ``LLMClient`` is ever
built and no request is ever sent.

Two deliberate conventions:

* every orchestrator is handed a caller-supplied ``EventBus``, which is what
  suppresses the ``RichReporter``; assertions read the same event stream a web
  dashboard would;
* nothing is asserted about a private helper unless that helper's behaviour is
  what makes a guarantee in this file true (``_is_done``, ``_get_tasks``,
  ``_execution_batches``, ``_original_goal``, ``_stall_cycles``).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from furrow.agents.worker import WorkerResult
from furrow.config import Plan, TaskModel, TaskStatus, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.events import EventType
from furrow.core.orchestrator import (
    EMPTY_PLAN_REASON,
    MIN_STALL_CYCLES,
    REPAIR_CONTEXT_HEADER,
    Orchestrator,
)
from furrow.errors import StructuredOutputError

#: A goal whose exact bytes matter: irregular spacing, a blank line, an
#: indented line, a non-ASCII character and trailing whitespace. Byte-identity
#: assertions against it catch any silent rewrite of the user's intent.
EXACT_GOAL = "Refactor the parser\n\nKeep  the  odd  spacing and the emoji \U0001f415\n   indented line exactly.\n  "

#: Verdict the stub tester returns unless a test says otherwise.
PASSING_TESTS = TestResult(passed=True, summary="42 passed", failures=[])

#: A failing verdict carrying real failure text, which the loop must keep.
FAILING_TESTS = TestResult(
    passed=False,
    summary="1 failed, 41 passed",
    failures=["tests/test_parser.py::test_emoji - AssertionError: emoji mangled"],
)


# ── Fakes ──────────────────────────────────────────────────────────────────


def make_task(
    task_id: str,
    *,
    files: Sequence[str] = (),
    dependencies: Sequence[str] = (),
    description: str | None = None,
) -> TaskModel:
    """A pending task. Ids are explicit because batching is about ids and files."""
    return TaskModel(
        id=task_id,
        description=f"work on {task_id}" if description is None else description,
        files=list(files),
        dependencies=list(dependencies),
    )


def make_plan(*tasks: TaskModel, rationale: str = "stub plan") -> Plan:
    """A plan over ``tasks``; with no tasks it is the loop's "finished" signal."""
    return Plan(tasks=list(tasks), rationale=rationale)


class ConcurrencyProbe:
    """Entry/exit instrumentation for the worker stub.

    ``limit`` is the bound the loop is supposed to respect: every entry beyond
    it is recorded in :attr:`violations`, so a runaway ``asyncio.gather`` is
    caught even if the peak bookkeeping were wrong.
    """

    def __init__(self, limit: int = 0) -> None:
        self.limit = limit
        self.active = 0
        self.peak = 0
        self.entered: list[str] = []
        self.exited: list[str] = []
        self.violations: list[tuple[str, int]] = []

    def enter(self, task_id: str) -> None:
        """Record a worker entering its fake body."""
        self.active += 1
        self.entered.append(task_id)
        self.peak = max(self.peak, self.active)
        if self.limit and self.active > self.limit:
            self.violations.append((task_id, self.active))

    def exit(self, task_id: str) -> None:
        """Record a worker leaving its fake body."""
        self.active -= 1
        self.exited.append(task_id)


@dataclass
class RecordedAgents:
    """Everything the stubs observed, for assertions."""

    goals: list[str] = field(default_factory=list)
    worker_task_ids: list[str] = field(default_factory=list)
    tester_calls: list[tuple[str, list[str]]] = field(default_factory=list)
    probe: ConcurrencyProbe = field(default_factory=ConcurrencyProbe)


WorkerChoice = WorkerResult | Mapping[str, WorkerResult] | Callable[[TaskModel], WorkerResult]


def install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plans: Sequence[Plan | BaseException] = (),
    planner_error: BaseException | None = None,
    worker: WorkerChoice | None = None,
    tests: TestResult | None = None,
    on_test: Callable[[], None] | None = None,
    work: float = 0.0,
    limit: int = 0,
) -> RecordedAgents:
    """Replace the orchestrator's three agents with offline recording stubs.

    ``plans`` is the planner script: entry *n* is handed back for cycle *n+1*
    (the last entry repeats forever) and a fresh deep copy each time, because
    the loop mutates task status in place and a re-used object would carry the
    previous cycle's state. An entry that is an exception instance is raised
    instead; ``planner_error`` raises on every call.

    ``worker`` may be one :class:`WorkerResult` for every task, a mapping from
    task id to result, or a callable taking the task. Each worker body sleeps
    ``work`` seconds with :func:`asyncio.sleep` so real concurrency is
    observable and so measured durations are genuinely non-zero; ``limit``
    bounds what :class:`ConcurrencyProbe` accepts.
    """
    recorded = RecordedAgents(probe=ConcurrencyProbe(limit=limit))
    script = list(plans)
    default_plan = script[-1] if script else make_plan()
    default_worker = WorkerResult(ok=True, summary="applied") if worker is None else worker
    verdict = PASSING_TESTS if tests is None else tests

    def choose_result(task: TaskModel) -> WorkerResult:
        """Resolve the result configured for ``task``."""
        if isinstance(default_worker, WorkerResult):
            return default_worker
        if isinstance(default_worker, Mapping):
            return default_worker.get(task.id, WorkerResult(ok=True, summary="applied"))
        return default_worker(task)

    class PlannerStub:
        """Stands in for :class:`~furrow.agents.planner.PlannerAgent`."""

        def __init__(self, client: Any = None, *, settings: Any = None) -> None:
            self.client = client
            self.settings = settings

        async def plan(self, goal: str, *, max_tasks: int | None = None) -> Plan:
            """Return the scripted plan (or raise the scripted failure)."""
            recorded.goals.append(goal)
            if planner_error is not None:
                raise planner_error
            response: Plan | BaseException = (
                script[min(len(recorded.goals) - 1, len(script) - 1)] if script else default_plan
            )
            if isinstance(response, BaseException):
                raise response
            return response.model_copy(deep=True)

    class WorkerStub:
        """Stands in for :class:`~furrow.agents.worker.WorkerAgent`."""

        def __init__(
            self,
            task: TaskModel,
            client: Any = None,
            *,
            settings: Any = None,
            tools: Any = None,
            attempt: int = 1,
            feedback: str = "",
        ) -> None:
            self.task = task
            self.client = client
            self.settings = settings
            self.tools = tools
            self.attempt = attempt
            self.feedback = feedback

        async def run(self) -> WorkerResult:
            """Simulate the worker's own work, then report the configured result."""
            recorded.worker_task_ids.append(self.task.id)
            recorded.probe.enter(self.task.id)
            try:
                await asyncio.sleep(work)
            finally:
                recorded.probe.exit(self.task.id)
            return choose_result(self.task)

    class TesterStub:
        """Stands in for :class:`~furrow.agents.tester.TesterAgent`."""

        def __init__(self, client: Any = None, *, settings: Any = None, workspace: Any = None) -> None:
            self.client = client
            self.settings = settings
            self.workspace = workspace

        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            """Record the call, run the optional hook, return the verdict."""
            recorded.tester_calls.append((goal, [task.id for task in tasks]))
            if on_test is not None:
                on_test()
            return verdict.model_copy(deep=True)

    monkeypatch.setattr(orchestrator_module, "PlannerAgent", PlannerStub)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", WorkerStub)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", TesterStub)
    return recorded


async def run_loop(orchestrator: Orchestrator) -> None:
    """``run()`` then ``aclose()``, the way the CLI drives the loop."""
    await orchestrator.run()
    await orchestrator.aclose()


# ── The loop itself ────────────────────────────────────────────────────────


@pytest.mark.usefixtures("agent_env")
async def test_loop_runs_every_cycle_in_the_budget(monkeypatch, make_settings, bus, collector, fake_client):
    """HEADLINE: a planner that always has work gets every cycle in the budget.

    This is the regression the whole product rests on. When ``_get_tasks()``
    returned ``[]`` the run computed ``completed >= len(tasks)`` as ``0 >= 0``,
    ``_is_done()`` answered True on the first call, the loop halted after one
    cycle and printed "Goal complete. Halting." — three cycles of planning,
    execution and testing never happened.
    """
    settings = make_settings(max_cycles=3, max_parallel_tasks=2)
    plan = make_plan(make_task("1", files=("a.py",)), make_task("2", files=("b.py",)))
    recorded = install_fakes(
        monkeypatch,
        plans=[plan],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=PASSING_TESTS,
    )
    orchestrator = Orchestrator(
        "keep going until the goal is done",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )

    await run_loop(orchestrator)

    assert orchestrator.cycles == 3, f"budget of 3 cycles, ran {orchestrator.cycles}"
    assert len(recorded.goals) == 3, "the planner must be consulted once per cycle"
    assert [record.cycle for record in orchestrator.history] == [1, 2, 3]
    assert all(record.completed == 2 for record in orchestrator.history)
    assert len(recorded.worker_task_ids) == 6, "two tasks executed in each of the three cycles"
    assert len(recorded.tester_calls) == 3, "the suite is run at the end of every cycle"
    assert [event.cycle for event in collector.of_type(EventType.CYCLE_STARTED)] == [1, 2, 3]

    completion = collector.of_type(EventType.RUN_COMPLETED)[0]
    assert completion.data["cycles"] == 3
    assert "reached the max_cycles budget of 3" in completion.data["reason"]
    assert collector.of_type(EventType.RUN_FAILED) == []


@pytest.mark.usefixtures("agent_env")
async def test_is_done_is_false_while_work_remains(monkeypatch, make_settings, bus, fake_client):
    """A cycle with outstanding work and a failing suite is NOT the end of a run.

    Every task in the cycle completed, the tests failed — exactly the mid-run
    state the loop must continue from.
    """
    settings = make_settings(max_cycles=0, max_parallel_tasks=2)  # 0 means unlimited
    recorded = install_fakes(
        monkeypatch,
        plans=[make_plan(make_task("1"), make_task("2"))],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=FAILING_TESTS,
    )
    orchestrator = Orchestrator(
        "never finished",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )
    fresh = Orchestrator("brand new", client=fake_client(settings), bus=bus, settings=settings)
    assert fresh._is_done() is False
    assert fresh.plan is None

    orchestrator._cycles = 1
    record = await orchestrator._cycle()

    assert record.completed == 2, "both planned tasks completed"
    assert record.tests_passed is False
    assert orchestrator._stall_cycles == 0, "completed work is progress"
    assert orchestrator._is_done() is False, "work remains, so the run must continue"
    assert len(recorded.goals) == 1


@pytest.mark.usefixtures("agent_env")
async def test_empty_plan_from_the_planner_ends_the_run(monkeypatch, make_settings, bus, collector, fake_client):
    """The planner's empty plan — not a task count — is the completion signal."""
    settings = make_settings(max_cycles=5, max_parallel_tasks=2)
    recorded = install_fakes(
        monkeypatch,
        plans=[make_plan(make_task("1")), make_plan()],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=FAILING_TESTS,
    )
    orchestrator = Orchestrator("finish me", client=fake_client(settings), bus=bus, settings=settings)

    await run_loop(orchestrator)

    assert orchestrator.cycles == 2, "the run ends on the empty plan, not at max_cycles=5"
    assert len(recorded.goals) == 2
    assert len(recorded.worker_task_ids) == 1, "nothing is executed for an empty plan"
    assert orchestrator.plan is not None and orchestrator.plan.is_empty
    assert orchestrator.history[-1].empty_plan is True
    assert orchestrator._is_done() is True
    assert orchestrator._termination_reason() == EMPTY_PLAN_REASON

    completion = collector.of_type(EventType.RUN_COMPLETED)[0]
    assert completion.data["reason"] == EMPTY_PLAN_REASON
    messages = [event.data.get("message", "") for event in collector.of_type(EventType.LOG)]
    assert any(EMPTY_PLAN_REASON in message for message in messages)


@pytest.mark.usefixtures("agent_env")
async def test_is_done_stops_on_a_stop_request(make_settings, bus):
    """A stop request wins over every other termination reason, reason included."""
    settings = make_settings(max_cycles=2, max_parallel_tasks=2)
    orchestrator = Orchestrator("stop me", bus=bus, settings=settings)

    assert orchestrator.stopped is False
    assert orchestrator._is_done() is False

    orchestrator.request_stop("engineer hit ctrl-c")

    assert orchestrator.stopped is True
    assert orchestrator._is_done() is True
    assert orchestrator._termination_reason() == "engineer hit ctrl-c"

    orchestrator._cycles = 2  # budget spent too: the stop reason still wins
    assert orchestrator._cycle_budget_reached() is True
    assert orchestrator._termination_reason() == "engineer hit ctrl-c"


@pytest.mark.usefixtures("agent_env")
async def test_is_done_stops_at_the_cycle_budget(make_settings, bus):
    """``max_cycles`` is a hard budget, and it is inclusive of the last cycle."""
    settings = make_settings(max_cycles=2, max_parallel_tasks=2)
    orchestrator = Orchestrator("budget me", bus=bus, settings=settings)

    orchestrator._cycles = 1
    assert orchestrator._cycle_budget_reached() is False
    assert orchestrator._is_done() is False

    orchestrator._cycles = 2
    assert orchestrator._cycle_budget_reached() is True
    assert orchestrator._is_done() is True
    assert orchestrator._termination_reason() == "reached the max_cycles budget of 2"


@pytest.mark.usefixtures("agent_env")
async def test_is_done_stops_on_an_empty_plan(make_settings, bus):
    """An empty stored plan is a completion signal even before the next cycle."""
    settings = make_settings(max_cycles=5, max_parallel_tasks=2)
    orchestrator = Orchestrator("nothing left", bus=bus, settings=settings)

    orchestrator._plan = make_plan()

    assert orchestrator._is_done() is True
    assert orchestrator._termination_reason() == EMPTY_PLAN_REASON


@pytest.mark.usefixtures("agent_env")
async def test_is_done_stops_at_the_stall_threshold(make_settings, bus):
    """Fruitless cycles end the run; the threshold has a floor of three."""
    settings = make_settings(max_cycles=0, max_parallel_tasks=2)
    orchestrator = Orchestrator("going nowhere", bus=bus, settings=settings)
    assert orchestrator._stall_threshold == MIN_STALL_CYCLES == 3

    orchestrator._stall_cycles = 2
    assert orchestrator._is_done() is False

    orchestrator._stall_cycles = 3
    assert orchestrator._is_done() is True
    reason = orchestrator._termination_reason()
    assert reason.startswith("stalled: 3 consecutive cycle(s) completed no new task")
    assert "threshold 3" in reason


@pytest.mark.usefixtures("agent_env")
async def test_get_tasks_is_empty_before_a_plan_and_holds_it_afterwards(
    monkeypatch, make_settings, bus, fake_client
):
    """``_get_tasks()`` is empty only until a plan exists, then it is the plan's.

    The regression this file exists for: this helper returning ``[]`` made the
    loop compare ``completed`` against zero and declare victory on cycle one.
    """
    settings = make_settings(max_cycles=1, max_parallel_tasks=2)
    install_fakes(
        monkeypatch,
        plans=[make_plan(make_task("1", files=("a.py",)), make_task("2", files=("b.py",)))],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=PASSING_TESTS,
    )
    orchestrator = Orchestrator(
        "collect the tasks",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )

    assert orchestrator.plan is None
    assert orchestrator._get_tasks() == []

    await run_loop(orchestrator)

    tasks = orchestrator._get_tasks()
    assert orchestrator.plan is not None
    assert [task.id for task in tasks] == ["1", "2"]
    assert tasks == orchestrator.plan.tasks
    assert tasks is not orchestrator.plan.tasks, "callers get their own list"
    assert [task.status for task in tasks] == [TaskStatus.COMPLETED, TaskStatus.COMPLETED]


# ── Parallelism ────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("agent_env")
async def test_max_parallel_tasks_is_a_real_bound(monkeypatch, make_settings, bus, collector, fake_client):
    """``max_parallel_tasks=2`` means at most two workers in flight, ever.

    An unbounded ``asyncio.gather`` over a batch of five would otherwise open
    five workers at once: the limit would be a comment rather than a bound.
    """
    settings = make_settings(max_parallel_tasks=2, max_cycles=1)
    tasks = [make_task(str(index), files=(f"src/mod_{index}.py",)) for index in range(1, 6)]
    recorded = install_fakes(
        monkeypatch,
        plans=[make_plan(*tasks)],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=PASSING_TESTS,
        work=0.02,
        limit=2,
    )
    orchestrator = Orchestrator(
        "five independent tasks",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )

    batches = await orchestrator._execution_batches(tasks)
    assert len(batches) == 1, "all five are independent, so they belong to one batch"
    assert len(batches[0]) == 5

    await run_loop(orchestrator)

    assert sorted(recorded.probe.entered) == ["1", "2", "3", "4", "5"]
    assert recorded.probe.peak == 2, f"expected 2 concurrent workers, observed {recorded.probe.peak}"
    assert recorded.probe.peak <= 2
    assert recorded.probe.violations == [], "a worker was in flight past the limit"
    assert recorded.probe.active == 0, "every worker exited"
    assert len(recorded.probe.exited) == 5


@pytest.mark.usefixtures("agent_env")
async def test_independent_non_overlapping_tasks_share_one_batch(monkeypatch, make_settings, bus, fake_client):
    """Nothing constrains them, so everything runs at once."""
    settings = make_settings(max_parallel_tasks=4, max_cycles=1)
    tasks = [
        make_task("1", files=("a.py",)),
        make_task("2", files=("b.py",)),
        make_task("3", files=("c.py",)),
    ]
    orchestrator = Orchestrator("batch me", bus=bus, settings=settings)

    batches = await orchestrator._execution_batches(tasks)

    assert len(batches) == 1
    assert [task.id for task in batches[0]] == ["1", "2", "3"]


@pytest.mark.usefixtures("agent_env")
async def test_tasks_sharing_a_file_are_separated(monkeypatch, make_settings, bus, collector, fake_client):
    """Declared files are soft mutual exclusion, even without a dependency edge."""
    settings = make_settings(max_parallel_tasks=4, max_cycles=1)
    tasks = [
        make_task("1", files=("shared.py",)),
        make_task("2", files=("shared.py",)),
        make_task("3", files=("other.py",)),
    ]
    orchestrator = Orchestrator("serialise me", bus=bus, settings=settings)

    batches = await orchestrator._execution_batches(tasks)
    batched = [[task.id for task in batch] for batch in batches]

    flat = [task_id for batch in batched for task_id in batch]
    assert sorted(flat) == ["1", "2", "3"], "every task is scheduled exactly once"
    for batch in batched:
        assert not {"1", "2"} <= set(batch), f"both editors of shared.py ran together in {batch}"
    assert len(batches) >= 2


@pytest.mark.usefixtures("agent_env")
async def test_dependent_task_lands_in_a_later_batch(monkeypatch, make_settings, bus, fake_client):
    """A dependency is a hard ordering constraint; everything else packs greedily."""
    settings = make_settings(max_parallel_tasks=4, max_cycles=1)
    tasks = [
        make_task("1", files=("a.py",)),
        make_task("3", files=("c.py",)),
        make_task("2", files=("b.py",), dependencies=("1",)),
    ]
    orchestrator = Orchestrator("order me", bus=bus, settings=settings)

    batches = await orchestrator._execution_batches(tasks)
    batched = [[task.id for task in batch] for batch in batches]

    assert batched[0] == ["1", "3"], "the independent tasks pack into the first batch"
    assert batched[-1] == ["2"]
    dependent_batch = next(index for index, batch in enumerate(batched) if "2" in batch)
    assert dependent_batch > 0, "a dependent task must not run in the first batch"
    assert "1" not in batched[dependent_batch], "a task ran beside its dependency"


@pytest.mark.usefixtures("agent_env")
async def test_dependency_cycle_neither_hangs_nor_pairs_a_task_with_its_dependency(
    monkeypatch, make_settings, bus, collector, fake_client
):
    """A malformed plan must still terminate and still not run a task beside its own dependency."""
    settings = make_settings(max_parallel_tasks=4, max_cycles=1)
    tasks = [
        make_task("1", files=("a.py",), dependencies=("2",)),
        make_task("2", files=("b.py",), dependencies=("1",)),
    ]
    orchestrator = Orchestrator("cycle me", bus=bus, settings=settings)

    batches = await asyncio.wait_for(orchestrator._execution_batches(tasks), timeout=5)

    assert sorted(task.id for batch in batches for task in batch) == ["1", "2"]
    for batch in batches:
        assert len(batch) == 1, f"a task was paired with its dependency: {[t.id for t in batch]}"
    messages = [event.data.get("message", "") for event in collector.of_type(EventType.LOG)]
    assert any("dependency cycle" in message for message in messages), messages


@pytest.mark.usefixtures("agent_env")
async def test_tasks_that_declare_no_files_are_not_serialised(
    monkeypatch, make_settings, bus, fake_client
):
    """Two tasks with ``files=[]`` overlap on nothing, so they are not serialised."""
    settings = make_settings(max_parallel_tasks=2, max_cycles=1)
    tasks = [make_task("1"), make_task("2")]
    orchestrator = Orchestrator("no files", bus=bus, settings=settings)

    batches = await orchestrator._execution_batches(tasks)

    assert len(batches) == 1
    assert [task.id for task in batches[0]] == ["1", "2"]


# ── The goal survives ──────────────────────────────────────────────────────


@pytest.mark.usefixtures("agent_env")
async def test_original_goal_survives_a_failing_test_cycle(monkeypatch, make_settings, bus, fake_client):
    """A failing suite adds context; it must never replace what the user asked for.

    The bug this guards: the goal was overwritten with the joined failure list,
    so the next cycle planned repairs to test failures instead of the feature
    the user requested — the intent was gone after one cycle.
    """
    settings = make_settings(max_cycles=2, max_parallel_tasks=2)
    recorded = install_fakes(
        monkeypatch,
        plans=[make_plan(make_task("1")), make_plan(make_task("2"))],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=FAILING_TESTS,
    )
    orchestrator = Orchestrator(EXACT_GOAL, client=fake_client(settings), bus=bus, settings=settings)

    await run_loop(orchestrator)

    assert orchestrator._original_goal.encode() == EXACT_GOAL.encode(), "the goal was rewritten"
    assert orchestrator.original_goal == EXACT_GOAL
    assert orchestrator.cycles == 2, "a failing cycle must not end the run"

    first_goal, second_goal = recorded.goals
    assert first_goal == EXACT_GOAL, "cycle 1 plans from the goal, verbatim"
    assert second_goal.startswith(EXACT_GOAL), "cycle 2 still leads with the original goal"
    assert second_goal != FAILING_TESTS.failures[0]
    assert REPAIR_CONTEXT_HEADER in second_goal
    assert FAILING_TESTS.failures[0] in second_goal, "the failure text is appended, not substituted"
    assert orchestrator.repair_context, "the repair context is accumulated for the next cycle"


# ── Failure handling ───────────────────────────────────────────────────────


@pytest.mark.usefixtures("agent_env")
async def test_planner_structured_output_error_stalls_and_stops_the_run(
    monkeypatch, make_settings, bus, collector, fake_client
):
    """An unparseable plan never escapes ``run()`` as a traceback.

    It counts as a fruitless cycle, and the stall rule — not an exception —
    decides when to give up.
    """
    settings = make_settings(max_cycles=0, max_parallel_tasks=2)  # unlimited: the stall rule must stop it
    recorded = install_fakes(
        monkeypatch,
        planner_error=StructuredOutputError(
            "planner did not return a valid plan after one retry: not JSON"
        ),
        worker=WorkerResult(ok=True, summary="applied"),
        tests=PASSING_TESTS,
    )
    orchestrator = Orchestrator(
        "unplannable",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )

    await asyncio.wait_for(run_loop(orchestrator), timeout=10)

    assert orchestrator._stall_cycles == orchestrator._stall_threshold == 3
    assert orchestrator.cycles == 3, "three fruitless cycles, then the stall rule stops the run"
    assert orchestrator.plan is None, "no plan was ever produced"
    assert recorded.worker_task_ids == [], "no plan means no execution"
    assert recorded.tester_calls == []
    assert len(orchestrator.history) == 3
    assert all("not JSON" in record.planner_error for record in orchestrator.history)

    assert collector.of_type(EventType.RUN_FAILED) == []
    completion = collector.of_type(EventType.RUN_COMPLETED)[0]
    assert completion.data["reason"].startswith("stalled: 3 consecutive cycle(s)")
    assert len(collector.of_type(EventType.CYCLE_COMPLETED)) == 3


@pytest.mark.usefixtures("agent_env")
async def test_failed_task_is_failed_and_its_dependent_is_skipped(
    monkeypatch, make_settings, bus, collector, fake_client
):
    """A failure fails its own task only; dependants are skipped, not attempted."""
    settings = make_settings(max_cycles=1, max_parallel_tasks=2)
    recorded = install_fakes(
        monkeypatch,
        plans=[
            make_plan(
                make_task("1", files=("a.py",)),
                make_task("2", files=("b.py",), dependencies=("1",)),
            )
        ],
        worker={"1": WorkerResult(ok=False, error="could not apply the patch")},
        tests=PASSING_TESTS,
    )
    orchestrator = Orchestrator(
        "fail one task",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )

    await run_loop(orchestrator)

    tasks = {task.id: task for task in orchestrator._get_tasks()}
    assert tasks["1"].status is TaskStatus.FAILED
    assert tasks["1"].error == "could not apply the patch"
    assert tasks["2"].status is TaskStatus.SKIPPED
    assert "1" in (tasks["2"].error or "")
    assert recorded.worker_task_ids == ["1"], "the dependent task is never handed to a worker"

    failures = collector.of_type(EventType.TASK_FAILED)
    assert [event.data["task_id"] for event in failures] == ["1"]
    assert failures[0].data["error"] == "could not apply the patch"
    skips = collector.of_type(EventType.TASK_SKIPPED)
    assert [event.data["task_id"] for event in skips] == ["2"]
    assert "blocked by unfinished dependency" in skips[0].data["reason"]
    assert collector.of_type(EventType.TASK_COMPLETED) == []

    record = orchestrator.history[-1]
    assert (record.failed, record.skipped, record.completed) == (1, 1, 0)


@pytest.mark.usefixtures("agent_env")
async def test_request_stop_reason_reaches_run_completed(monkeypatch, make_settings, bus, collector, fake_client):
    """Stopping mid-run finishes the cycle cleanly and reports the caller's reason."""
    settings = make_settings(max_cycles=5, max_parallel_tasks=2)
    orchestrator = Orchestrator(
        "stop me midway",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )
    recorded = install_fakes(
        monkeypatch,
        plans=[make_plan(make_task("1"))],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=PASSING_TESTS,
        on_test=lambda: orchestrator.request_stop("user pressed stop"),
    )

    await run_loop(orchestrator)

    assert orchestrator.stopped is True
    assert orchestrator.cycles == 1
    assert len(recorded.tester_calls) == 1
    assert len(collector.of_type(EventType.CYCLE_STARTED)) == 1
    completion = collector.of_type(EventType.RUN_COMPLETED)[0]
    assert completion.data == {"cycles": 1, "reason": "user pressed stop"}


@pytest.mark.usefixtures("agent_env")
async def test_request_stop_without_a_reason_uses_the_default_reason(
    monkeypatch, make_settings, bus, collector, fake_client
):
    """A bare ``request_stop()`` before the loop still halts it, with a reason."""
    settings = make_settings(max_cycles=5, max_parallel_tasks=2)
    recorded = install_fakes(
        monkeypatch,
        plans=[make_plan(make_task("1"))],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=PASSING_TESTS,
    )
    orchestrator = Orchestrator(
        "stop me now",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )
    orchestrator.request_stop()

    await run_loop(orchestrator)

    assert orchestrator.cycles == 0
    assert recorded.goals == [], "a stop request means no cycle is started"
    completion = collector.of_type(EventType.RUN_COMPLETED)[0]
    assert completion.data == {"cycles": 0, "reason": "stop requested"}


# ── The event stream ───────────────────────────────────────────────────────


@pytest.mark.usefixtures("agent_env")
async def test_successful_cycle_emits_the_expected_event_sequence(
    monkeypatch, make_settings, bus, collector, fake_client
):
    """One successful cycle, one task: the stream a dashboard renders, in order."""
    settings = make_settings(max_cycles=1, max_parallel_tasks=2)
    install_fakes(
        monkeypatch,
        plans=[make_plan(make_task("1", description="write the module", files=("a.py",)))],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=PASSING_TESTS,
    )
    orchestrator = Orchestrator(
        "one clean cycle",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )

    await run_loop(orchestrator)

    assert collector.types() == [
        "run.started",
        "cycle.started",
        "plan.created",
        "task.started",
        "task.completed",
        "tests.completed",
        "cycle.completed",
        "run.completed",
    ]
    assert collector.of_type(EventType.TESTS_COMPLETED)[0].data == {
        "passed": True,
        "summary": PASSING_TESTS.summary,
        "failures": [],
    }
    assert collector.of_type(EventType.TASK_STARTED)[0].data["description"] == "write the module"
    assert collector.of_type(EventType.CYCLE_COMPLETED)[0].data == {"passed": True}


@pytest.mark.usefixtures("agent_env")
async def test_plan_created_carries_its_own_cycle_and_a_json_payload(
    monkeypatch, make_settings, bus, collector, fake_client
):
    """Every plan belongs to the cycle that produced it.

    The bug this guards: ``plan_created`` took no ``cycle``, so its events all
    carried the default 0 and anything grouping the stream by cycle
    attributed every plan to cycle 0.
    """
    settings = make_settings(max_cycles=2, max_parallel_tasks=2)
    install_fakes(
        monkeypatch,
        plans=[
            make_plan(make_task("1", files=("first.py",)), rationale="first pass"),
            make_plan(make_task("1", files=("second.py",)), rationale="second pass"),
        ],
        worker=WorkerResult(ok=True, summary="applied"),
        tests=PASSING_TESTS,
    )
    orchestrator = Orchestrator(
        "attribute my plans",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )

    await run_loop(orchestrator)

    plan_events = collector.of_type(EventType.PLAN_CREATED)
    assert len(plan_events) == 2
    assert [event.cycle for event in plan_events] == [1, 2], "plans were attributed to cycle 0"
    assert [event.data["plan"]["rationale"] for event in plan_events] == ["first pass", "second pass"]
    for event in plan_events:
        payload = event.data["plan"]
        assert payload["tasks"][0]["status"] == "pending"
        assert json.loads(json.dumps(event.data)) == event.data, "payload must be JSON-safe"


@pytest.mark.usefixtures("agent_env")
async def test_task_events_carry_real_durations(monkeypatch, make_settings, bus, collector, fake_client):
    """Durations are measured, not the ``0.0`` default of the event factories."""
    settings = make_settings(max_cycles=1, max_parallel_tasks=2)
    install_fakes(
        monkeypatch,
        plans=[
            make_plan(
                make_task("1", files=("a.py",)),
                make_task("2", files=("b.py",)),
            )
        ],
        worker={
            "1": WorkerResult(ok=True, summary="applied"),
            "2": WorkerResult(ok=False, error="patch did not apply"),
        },
        tests=PASSING_TESTS,
        work=0.02,
    )
    orchestrator = Orchestrator(
        "time me",
        client=fake_client(settings),
        bus=bus,
        settings=settings,
    )

    await run_loop(orchestrator)

    finished = collector.of_type(EventType.TASK_COMPLETED) + collector.of_type(EventType.TASK_FAILED)
    assert len(finished) == 2
    for event in finished:
        duration = event.data["duration"]
        assert isinstance(duration, float)
        assert duration > 0.0, f"default duration leaked into {event.data}"
        assert duration >= 0.01, f"duration {duration} does not cover the worker's own time"


# ── Shutdown ───────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("agent_env")
async def test_aclose_is_idempotent_and_leaves_a_supplied_client_open(make_settings, bus, fake_client):
    """Two ``aclose()`` calls are safe, and the caller's client survives both."""
    settings = make_settings(max_cycles=1)
    client = fake_client(settings)
    orchestrator = Orchestrator("close me", client=client, bus=bus, settings=settings)

    await orchestrator.aclose()
    await orchestrator.aclose()

    assert client.closed is False, "a caller-supplied client belongs to the caller"


@pytest.mark.usefixtures("agent_env")
async def test_aclose_never_builds_a_client_when_none_was_supplied(make_settings, bus):
    """Shutting down an orchestrator that never ran needs no client at all."""
    settings = make_settings(max_cycles=1)
    orchestrator = Orchestrator("never ran", bus=bus, settings=settings)

    assert orchestrator.client is None

    await orchestrator.aclose()

    assert orchestrator.client is None, "aclose must not construct an LLM client"
