"""Behavioural tests for the ``furrow`` orchestrator loop and its event model.

Each test name says which regression it guards. The headline one is
``TestLoopRunsMultipleCycles::test_loop_runs_more_than_one_cycle``: an earlier
revision had ``_get_tasks()`` return ``[]``, which made ``_is_done()`` trivially
true and therefore always halted the loop after a single cycle.

Constraints honoured throughout:

* no network, no API key and no real test runner -- the LLM is the canned
  ``FakeLLMClient`` double and ``TesterAgent.run`` is replaced by a scripted stub;
* every await is bounded by ``asyncio.wait_for`` through the ``with_timeout``
  fixture, so a scheduler regression fails instead of hanging the suite;
* settings are per-test copies (``tmp_settings``); the global
  ``furrow.config.settings`` singleton is never mutated.
"""

from __future__ import annotations

import asyncio
import io
import json
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest
from rich.console import Console

import furrow.core.orchestrator as orchestrator_module
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.events import Event, EventType
from furrow.core.orchestrator import NO_PROGRESS_LIMIT, Orchestrator
from tests.conftest import FakeLLMClient

GOAL = "build the thing"

#: Upper bound for any single awaited operation inside a test.
TIMEOUT = 20.0

PASSING = TestResult(passed=True, summary="all green")
FAILING = TestResult(
    passed=False,
    summary="1 failed, 1 passed",
    failures=["FAILED tests/test_widget.py::test_build"],
)

TASK_ID_RE = re.compile(r"Task id:\s*(\S+)")


# --------------------------------------------------------------------- helpers


def make_task(task_id: str, dependencies: Sequence[str] | None = None) -> TaskModel:
    """Build a planned task. Ids are strings: pydantic will not coerce ints."""
    return TaskModel(
        id=task_id,
        description=f"do work {task_id}",
        dependencies=list(dependencies or []),
    )


def plan_payload(tasks: Sequence[TaskModel], rationale: str = "because") -> dict[str, Any]:
    """The raw JSON the planner model is expected to return for ``tasks``."""
    return {
        "tasks": [
            {
                "id": task.id,
                "description": task.description,
                "files": [],
                "dependencies": task.dependencies,
            }
            for task in tasks
        ],
        "rationale": rationale,
    }


def last_done(events: Sequence[Event]) -> Event:
    """Return the single ``DONE`` event, asserting there is exactly one."""
    dones = [event for event in events if event.type is EventType.DONE]
    assert len(dones) == 1, f"expected exactly one DONE event, got {len(dones)}"
    return dones[0]


def _raise_sync(event: Event) -> None:
    raise RuntimeError("sync callback exploded")


async def _raise_async(event: Event) -> None:
    raise RuntimeError("async callback exploded")


class ScriptedTester:
    """Stands in for ``TesterAgent.run`` so no test runner is ever launched."""

    def __init__(self, results: Sequence[TestResult]) -> None:
        self.results = list(results)
        self.calls = 0
        self.goals: list[str] = []
        self.seen_tasks: list[list[TaskModel]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.goals.append(goal)
        self.seen_tasks.append(list(tasks))
        result = self.results[min(self.calls, len(self.results) - 1)]
        self.calls += 1
        return result


class WorkerProbe:
    """Prompt hook recording worker order and peak parallelism, injecting failures.

    With ``expect_parallel`` set, the first N concurrent workers rendezvous on an
    ``asyncio.Event`` before any of them finishes. That is a deterministic,
    sleep-free way to prove the scheduler really overlaps work: a cap that is too
    high shows up as an ``order`` that never completes (surfaced as a timeout by
    ``with_timeout``) or as a peak larger than the cap.
    """

    def __init__(self, *, expect_parallel: int = 0, fail_ids: Sequence[str] = ()) -> None:
        self.order: list[str] = []
        self.peak = 0
        self.fail_ids = set(fail_ids)
        self._expect_parallel = expect_parallel
        self._inflight = 0
        self._arrivals = 0
        self._released: asyncio.Event | None = None

    async def __call__(self, prompt: str) -> None:
        match = TASK_ID_RE.search(prompt)
        task_id = match.group(1) if match else "?"
        if self._released is None:
            self._released = asyncio.Event()
        self._inflight += 1
        self._arrivals += 1
        self.peak = max(self.peak, self._inflight)
        if self._expect_parallel and self._arrivals >= self._expect_parallel:
            self._released.set()
        try:
            if self._expect_parallel:
                await self._released.wait()
            self.order.append(task_id)
            if task_id in self.fail_ids:
                raise RuntimeError(f"worker boom on task {task_id}")
        finally:
            self._inflight -= 1


@dataclass
class Harness:
    """An orchestrator plus the doubles it was wired to."""

    orchestrator: Orchestrator
    client: FakeLLMClient
    tester: ScriptedTester
    probe: WorkerProbe
    events: list[Event] = field(default_factory=list)
    plan_goals: list[str] = field(default_factory=list)
    plan_contexts: list[str | None] = field(default_factory=list)

    @property
    def plan_calls(self) -> int:
        """How many times the planner was consulted."""
        return len(self.plan_goals)


def _track_planner(orchestrator: Orchestrator, plan: Plan | None, harness: Harness) -> None:
    """Count and record planner calls; ``plan`` bypasses the agent with a fixed plan.

    Bypassing ``PlannerAgent`` also lifts its ``MAX_TASKS`` ceiling, which is what
    lets the scheduler tests drive plans that are larger than the planner would
    ever emit.
    """
    if plan is not None:

        async def stub_plan(goal: str, context: str | None = None) -> Plan:
            harness.plan_goals.append(goal)
            harness.plan_contexts.append(context)
            return plan.model_copy(deep=True)

        orchestrator.planner.plan = stub_plan  # type: ignore[method-assign]
        return

    original = orchestrator.planner.plan

    async def counted_plan(goal: str, context: str | None = None) -> Plan:
        harness.plan_goals.append(goal)
        harness.plan_contexts.append(context)
        return await original(goal, context=context)

    orchestrator.planner.plan = counted_plan  # type: ignore[method-assign]


# --------------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def quiet_console() -> Any:
    """Keep rich panels and spinners out of the captured test report."""
    original = orchestrator_module.console
    orchestrator_module.console = Console(file=io.StringIO(), width=100, force_terminal=False)
    yield
    orchestrator_module.console = original


@pytest.fixture
def with_timeout() -> Callable[..., Awaitable[Any]]:
    """Return a helper that runs a coroutine under ``asyncio.wait_for``."""

    async def _run(awaitable: Awaitable[Any], timeout: float = TIMEOUT) -> Any:
        return await asyncio.wait_for(awaitable, timeout=timeout)

    return _run


@pytest.fixture
def make_orch(tmp_settings: Callable[..., Settings], fake_llm_client: Callable[..., Any]) -> Any:
    """Return a factory building a :class:`Harness` wired to offline doubles."""

    def _make(
        *,
        tasks: Sequence[TaskModel] | None = None,
        rationale: str = "because",
        tester_results: Sequence[TestResult] | None = None,
        raw_plan: Plan | None = None,
        on_event: Callable[[Event], Any] | None = None,
        settings: Settings | None = None,
        probe: WorkerProbe | None = None,
        goal: str = GOAL,
    ) -> Harness:
        chosen = settings or tmp_settings()
        planned = [make_task("1")] if tasks is None else list(tasks)
        payload = plan_payload(planned, rationale)
        worker_probe = probe if probe is not None else WorkerProbe()
        client = fake_llm_client(
            settings=chosen, json_payload=payload, hook=worker_probe
        )
        tester = ScriptedTester([PASSING] if tester_results is None else list(tester_results))
        events: list[Event] = []
        orchestrator = Orchestrator(
            goal,
            client=client,
            settings=chosen,
            on_event=events.append if on_event is None else on_event,
        )
        orchestrator.tester = tester  # type: ignore[assignment]
        harness = Harness(
            orchestrator=orchestrator,
            client=client,
            tester=tester,
            probe=worker_probe,
            events=events,
        )
        _track_planner(orchestrator, raw_plan, harness)
        return harness

    return _make


# ------------------------------------------------------------ 1. the loop runs


class TestLoopRunsMultipleCycles:
    """The headline regression: the loop must keep cycling while work remains."""

    async def test_loop_runs_more_than_one_cycle(self, make_orch: Any, with_timeout: Any) -> None:
        """An unchanged plan with failing tests must not end the loop after cycle 1.

        Regression guard: ``_get_tasks()`` used to return ``[]``, so ``_is_done()``
        was trivially true and ``run()`` always halted after a single cycle.
        """
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=3))

        assert harness.orchestrator.cycles == 3
        assert harness.plan_calls == 3, "planner must be consulted once per cycle"
        assert harness.client.json_calls == 3
        # The pre-fix failure mode, stated explicitly.
        assert harness.orchestrator.cycles != 1
        assert harness.client.json_calls != 1
        # Failing tests keep work outstanding, so the loop is legitimately not done.
        assert harness.orchestrator.state()["done"] is False
        assert harness.tester.calls == 3

    async def test_one_cycle_start_event_per_cycle(self, make_orch: Any, with_timeout: Any) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=3))

        assert [event.type for event in harness.events].count(EventType.CYCLE_START) == 3

    @pytest.mark.parametrize("limit", [1, 2, 3])
    async def test_run_stops_at_the_cycle_limit(
        self, make_orch: Any, with_timeout: Any, limit: int
    ) -> None:
        """3 is the largest budget reachable here: from cycle 4 on, an unchanging
        plan trips the no-progress guard before the next cycle limit check (see
        ``TestNoProgressGuard``)."""
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=limit))

        assert harness.orchestrator.cycles == limit
        assert harness.plan_calls == limit
        assert f"cycle limit of {limit}" in last_done(harness.events).message.lower()


# --------------------------------------------------------- 2. the cycle budget


class TestCycleLimits:
    """``max_cycles`` defaults to 0 (unlimited) and ``run()`` can override it."""

    async def test_zero_max_cycles_is_unlimited(
        self, make_orch: Any, tmp_settings: Callable[..., Settings], with_timeout: Any
    ) -> None:
        """``max_cycles=0`` must reach another stop condition, not loop forever."""
        harness = make_orch(
            tasks=[make_task("1")],
            tester_results=[FAILING],
            settings=tmp_settings(max_cycles=0),
        )

        await with_timeout(harness.orchestrator.run())

        assert harness.orchestrator.cycles > 1, "0 means unlimited, not 'one cycle'"
        reason = last_done(harness.events).message.lower()
        assert "cycle limit" not in reason, "an unlimited run cannot stop on a cycle limit"

    async def test_run_max_cycles_overrides_the_setting(
        self, make_orch: Any, tmp_settings: Callable[..., Settings], with_timeout: Any
    ) -> None:
        harness = make_orch(
            tasks=[make_task("1")],
            tester_results=[FAILING],
            settings=tmp_settings(max_cycles=9),
        )

        await with_timeout(harness.orchestrator.run(max_cycles=2))

        assert harness.orchestrator.cycles == 2
        assert "cycle limit of 2" in last_done(harness.events).message.lower()

    async def test_setting_max_cycles_is_used_without_an_override(
        self, make_orch: Any, tmp_settings: Callable[..., Settings], with_timeout: Any
    ) -> None:
        harness = make_orch(
            tasks=[make_task("1")],
            tester_results=[FAILING],
            settings=tmp_settings(max_cycles=2),
        )

        await with_timeout(harness.orchestrator.run())

        assert harness.orchestrator.cycles == 2


# ------------------------------------------------------------ 3. request_stop()


class TestRequestStop:
    """``request_stop()`` ends the loop cleanly with a stop-flavoured reason."""

    async def test_request_stop_from_event_callback_ends_the_loop(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])
        orchestrator = harness.orchestrator

        def on_event(event: Event) -> None:
            harness.events.append(event)
            if event.type is EventType.CYCLE_END and event.cycle == 1:
                orchestrator.request_stop()

        orchestrator.on_event = on_event

        await with_timeout(orchestrator.run(max_cycles=10))

        assert orchestrator.cycles == 1
        reason = last_done(harness.events).message.lower()
        assert "stop" in reason
        assert "goal complete" not in reason
        assert orchestrator.state()["done"] is False


# ----------------------------------------------------------- 4/5/6. scheduling


class TestDependencyScheduling:
    """The wave scheduler orders by ``dependencies`` and always terminates."""

    async def test_dependency_chain_runs_in_order(self, make_orch: Any, with_timeout: Any) -> None:
        """``1 -> 2 -> 3`` must execute strictly in that order."""
        plan = Plan(
            tasks=[make_task("1"), make_task("2", ["1"]), make_task("3", ["2"])],
            rationale="chain",
        )
        harness = make_orch(raw_plan=plan, tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=1))

        assert harness.probe.order == ["1", "2", "3"]
        assert all(
            task.status == "completed" for task in harness.orchestrator.tasks.values()
        ), harness.orchestrator.tasks

    async def test_dependency_cycle_does_not_hang(self, make_orch: Any, with_timeout: Any) -> None:
        """A cyclic dependency graph is broken, not waited on forever."""
        plan = Plan(tasks=[make_task("1", ["2"]), make_task("2", ["1"])], rationale="cycle")
        harness = make_orch(raw_plan=plan, tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=1))

        assert sorted(harness.probe.order) == ["1", "2"]
        assert harness.orchestrator.cycles == 1

    async def test_task_with_missing_dependency_still_runs(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        """A dependency absent from the plan must not deadlock the task."""
        plan = Plan(tasks=[make_task("1", ["99"])], rationale="dangling dependency")
        harness = make_orch(raw_plan=plan, tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=1))

        assert harness.probe.order == ["1"]
        assert harness.orchestrator.tasks["1"].status == "completed"

    async def test_planner_drops_dependencies_it_cannot_resolve(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        """The same shape through the real planner: dangling deps are stripped."""
        harness = make_orch(tasks=[make_task("1", ["99"])], tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=1))

        assert harness.orchestrator.tasks["1"].dependencies == []
        assert harness.probe.order == ["1"]


# ------------------------------------------------------ 7. concurrency ceiling


class TestConcurrencyCap:
    """In-flight tasks never exceed ``settings.max_parallel_tasks``."""

    async def test_peak_parallelism_respects_a_cap_of_two(
        self, make_orch: Any, tmp_settings: Callable[..., Settings], with_timeout: Any
    ) -> None:
        plan = Plan(tasks=[make_task(str(n)) for n in range(1, 7)], rationale="six independent")
        probe = WorkerProbe(expect_parallel=2)
        harness = make_orch(
            raw_plan=plan,
            tester_results=[FAILING],
            settings=tmp_settings(max_parallel_tasks=2),
            probe=probe,
        )

        await with_timeout(harness.orchestrator.run(max_cycles=1), timeout=5.0)

        assert sorted(harness.probe.order) == ["1", "2", "3", "4", "5", "6"]
        assert probe.peak <= 2, f"peak in-flight tasks was {probe.peak}"
        assert probe.peak == 2, "the cap must not serialise independent tasks"

    async def test_peak_parallelism_respects_a_cap_of_three(
        self, make_orch: Any, tmp_settings: Callable[..., Settings], with_timeout: Any
    ) -> None:
        plan = Plan(tasks=[make_task(str(n)) for n in range(1, 6)], rationale="five independent")
        probe = WorkerProbe(expect_parallel=3)
        harness = make_orch(
            raw_plan=plan,
            tester_results=[FAILING],
            settings=tmp_settings(max_parallel_tasks=3),
            probe=probe,
        )

        await with_timeout(harness.orchestrator.run(max_cycles=1), timeout=5.0)

        assert sorted(harness.probe.order) == ["1", "2", "3", "4", "5"]
        assert probe.peak == 3


# ------------------------------------------------------------ 8. worker failure


class TestWorkerFailure:
    """A raising worker becomes a failed task; ``run()`` does not propagate."""

    async def test_failing_worker_is_recorded_and_loop_continues(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        plan = Plan(tasks=[make_task("1"), make_task("2")], rationale="one bad apple")
        harness = make_orch(
            raw_plan=plan,
            tester_results=[FAILING],
            probe=WorkerProbe(fail_ids=["2"]),
        )

        # Two cycles proves the failure did not escape and stop the loop.
        await with_timeout(harness.orchestrator.run(max_cycles=2))

        assert harness.orchestrator.cycles == 2
        bad = harness.orchestrator.tasks["2"]
        good = harness.orchestrator.tasks["1"]
        assert bad.status == "failed"
        assert "worker boom on task 2" in (bad.result or "")
        assert good.status == "completed"
        assert any(event.type is EventType.ERROR for event in harness.events)
        # The failure is handed back to the planner as context for the next cycle.
        assert any("worker boom" in (context or "") for context in harness.plan_contexts)


# -------------------------------------------------------------------- 9. events


class TestEvents:
    """Event ordering, serialisability and callback isolation."""

    async def test_event_sequence_for_one_happy_cycle(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[PASSING])

        await with_timeout(harness.orchestrator.run(max_cycles=1))

        assert [event.type for event in harness.events] == [
            EventType.CYCLE_START,
            EventType.PLAN,
            EventType.TASK_START,
            EventType.TASK_END,
            EventType.TEST_START,
            EventType.TEST_END,
            EventType.CYCLE_END,
            EventType.DONE,
        ]

    async def test_cycle_start_cycle_end_done_are_ordered(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=2))

        types = [event.type for event in harness.events]
        assert types[0] is EventType.CYCLE_START
        assert types[-1] is EventType.DONE
        assert types.index(EventType.CYCLE_START) < types.index(EventType.CYCLE_END)
        assert types.index(EventType.CYCLE_END) < types.index(EventType.DONE)

    async def test_every_event_is_json_serialisable(self, make_orch: Any, with_timeout: Any) -> None:
        """Covers the ERROR path too, by failing one worker and the tester."""
        plan = Plan(tasks=[make_task("1"), make_task("2")], rationale="mixed")
        harness = make_orch(
            raw_plan=plan,
            tester_results=[FAILING],
            probe=WorkerProbe(fail_ids=["2"]),
        )

        await with_timeout(harness.orchestrator.run(max_cycles=2))

        payloads = [json.loads(json.dumps(event.to_dict())) for event in harness.events]
        assert {payload["type"] for payload in payloads} >= {
            "cycle_start",
            "plan",
            "task_start",
            "task_end",
            "test_start",
            "test_end",
            "cycle_end",
            "error",
            "done",
        }

    async def test_done_event_carries_the_run_summary(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=2))

        done = last_done(harness.events)
        assert done.data["cycles"] == 2
        assert done.data["tasks"] == {"1": "completed"}
        assert done.data["test_failures"] == FAILING.failures
        assert done.data["stop_requested"] is False
        assert done.data["done"] is False
        assert done.data["no_progress_cycles"] >= 1

    @pytest.mark.parametrize("explode", [_raise_sync, _raise_async], ids=["sync", "async"])
    async def test_faulty_event_callback_does_not_break_the_loop(
        self, make_orch: Any, with_timeout: Any, explode: Callable[[Event], Any]
    ) -> None:
        harness = make_orch(
            tasks=[make_task("1")], tester_results=[FAILING], on_event=explode
        )

        await with_timeout(harness.orchestrator.run(max_cycles=2))

        assert harness.orchestrator.cycles == 2
        assert harness.plan_calls == 2
        assert harness.tester.calls == 2, "the tester must still run despite the callback"

    async def test_passing_tests_reach_done_despite_a_faulty_callback(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        """A passing suite ends the loop normally even when the callback raises."""
        harness = make_orch(
            tasks=[make_task("1")], tester_results=[PASSING], on_event=_raise_sync
        )

        await with_timeout(harness.orchestrator.run(max_cycles=5))

        assert harness.orchestrator.cycles == 1
        assert harness.orchestrator.state()["done"] is True


# ------------------------------------------------------------------- 10. state


class TestStateSnapshot:
    """``state()`` is the JSON-serialisable snapshot the web UI renders."""

    async def test_state_is_json_serialisable_with_the_documented_keys(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run(max_cycles=2))

        snapshot = harness.orchestrator.state()
        assert set(snapshot) == {"goal", "cycles", "done", "tasks", "history"}
        restored = json.loads(json.dumps(snapshot))
        assert restored["goal"] == GOAL
        assert restored["cycles"] == 2
        assert restored["done"] is False
        assert [task["id"] for task in restored["tasks"]] == ["1"]
        assert restored["tasks"][0]["status"] == "completed"
        assert len(restored["history"]) == 2

    def test_state_before_any_cycle_is_not_done(self, make_orch: Any) -> None:
        """No tasks *and* no plan yet must not read as 'goal complete'.

        This is the exact condition the old ``_get_tasks() -> []`` bug produced.
        """
        snapshot = make_orch().orchestrator.state()

        assert snapshot["cycles"] == 0
        assert snapshot["tasks"] == []
        assert snapshot["history"] == []
        assert snapshot["done"] is False


# ----------------------------------------------- 11. cycle failure containment


class TestCycleFailureContainment:
    """One bad cycle is reported and contained; the loop keeps going."""

    async def test_cycle_exception_emits_error_and_run_survives(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        async def exploding_plan(goal: str, context: str | None = None) -> Plan:
            harness.plan_goals.append(goal)
            raise RuntimeError("planner exploded")

        harness.orchestrator.planner.plan = exploding_plan  # type: ignore[method-assign]

        await with_timeout(harness.orchestrator.run(max_cycles=2))

        assert harness.orchestrator.cycles == 2
        errors = [event for event in harness.events if event.type is EventType.ERROR]
        assert any("planner exploded" in event.message for event in errors)
        assert [event.type for event in harness.events].count(EventType.CYCLE_END) == 2
        assert harness.client.text_calls == 0, "a failed cycle runs no workers"

    async def test_repeated_cycle_failures_still_terminate(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        """An unlimited run whose every cycle raises must still halt."""
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        async def exploding_plan(goal: str, context: str | None = None) -> Plan:
            harness.plan_goals.append(goal)
            raise RuntimeError("planner exploded")

        harness.orchestrator.planner.plan = exploding_plan  # type: ignore[method-assign]

        await with_timeout(harness.orchestrator.run())

        assert harness.orchestrator.cycles == NO_PROGRESS_LIMIT
        assert "no progress" in last_done(harness.events).message.lower()


# -------------------------------------------------------------- 12. cancellation


class TestCancellation:
    """``asyncio.CancelledError`` is never swallowed by ``run()``."""

    async def test_cancelled_error_from_a_cycle_propagates(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        async def cancelling_cycle() -> None:
            raise asyncio.CancelledError("cancelled mid-cycle")

        harness.orchestrator._cycle = cancelling_cycle  # type: ignore[method-assign]

        with pytest.raises(asyncio.CancelledError):
            await with_timeout(harness.orchestrator.run(max_cycles=3))

        assert harness.orchestrator.cycles == 1

    async def test_external_cancellation_is_not_swallowed(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[PASSING])
        started = asyncio.Event()

        async def slow_cycle() -> None:
            started.set()
            await asyncio.sleep(3600)

        harness.orchestrator._cycle = slow_cycle  # type: ignore[method-assign]
        runner = asyncio.create_task(harness.orchestrator.run())
        await with_timeout(started.wait())
        runner.cancel()

        with pytest.raises(asyncio.CancelledError):
            await with_timeout(runner)


# ------------------------------------------------------- 13. no-progress guard


class TestNoProgressGuard:
    """A plan that stops changing terminates instead of spinning forever."""

    async def test_unchanging_plan_stops_on_the_no_progress_guard(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING])

        await with_timeout(harness.orchestrator.run())

        assert harness.orchestrator.cycles == NO_PROGRESS_LIMIT + 1
        assert "no progress" in last_done(harness.events).message.lower()
        assert any(event.type is EventType.ERROR for event in harness.events)

    async def test_empty_plan_completes_the_goal_in_one_cycle(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        harness = make_orch(tasks=[], tester_results=[PASSING])

        await with_timeout(harness.orchestrator.run())

        assert harness.orchestrator.cycles == 1
        assert harness.client.text_calls == 0
        assert "goal complete" in last_done(harness.events).message.lower()


# ------------------------------------------------------ 14. goal preservation


class TestGoalPreservation:
    """The original goal survives cycles; only the planner sees the follow-up."""

    async def test_original_goal_survives_a_failing_test_cycle(
        self, make_orch: Any, with_timeout: Any
    ) -> None:
        """Regression guard: the goal used to be overwritten by the fix prompt."""
        harness = make_orch(tasks=[make_task("1")], tester_results=[FAILING], goal=GOAL)

        await with_timeout(harness.orchestrator.run(max_cycles=2))

        assert harness.orchestrator.goal == GOAL
        assert harness.orchestrator.state()["goal"] == GOAL
        assert harness.tester.goals == [GOAL, GOAL], "the tester always gets the raw goal"
        assert harness.plan_goals[0] == GOAL
        assert harness.plan_goals[1].startswith(GOAL)
        assert harness.plan_goals[1] != GOAL, "the failure follow-up must reach the planner"
        assert FAILING.failures[0] in harness.plan_goals[1]


# ------------------------------------------------------------- 15. event model


class TestEventModel:
    """``Event`` / ``EventType`` shape, independent of the loop."""

    def test_event_type_values_are_the_expected_strings(self) -> None:
        assert {member.value for member in EventType} == {
            "cycle_start",
            "plan",
            "task_start",
            "task_end",
            "test_start",
            "test_end",
            "cycle_end",
            "error",
            "done",
        }
        assert {member.name.lower() for member in EventType} == {member.value for member in EventType}

    def test_event_type_is_a_str_enum(self) -> None:
        assert issubclass(EventType, str)
        assert EventType.CYCLE_START == "cycle_start"
        assert EventType("plan") is EventType.PLAN
        assert EventType.PLAN.value == "plan"
        assert type(EventType.PLAN.value) is str

    def test_event_to_dict_shape(self) -> None:
        event = Event(type=EventType.PLAN, cycle=2, message="why", data={"k": [1, 2]})

        payload = event.to_dict()

        assert set(payload) == {"type", "cycle", "message", "data", "timestamp"}
        assert payload["type"] == "plan"
        assert payload["cycle"] == 2
        assert payload["message"] == "why"
        assert payload["data"] == {"k": [1, 2]}
        assert payload["timestamp"] == event.timestamp
        assert json.loads(json.dumps(payload))["data"] == {"k": [1, 2]}

    def test_event_defaults_and_per_instance_data(self) -> None:
        first = Event(type=EventType.DONE)
        second = Event(type=EventType.DONE)

        assert first.cycle == 0
        assert first.message == ""
        assert first.data == {}
        assert first.timestamp > 0

        first.data["only-mine"] = True
        assert second.data == {}, "Event.data must not be shared between instances"
