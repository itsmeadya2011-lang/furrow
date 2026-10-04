"""Offline unit tests for the orchestrator loop.

Everything here is fake: an ``LLMClient`` subclass returns canned JSON, and the
tester's subprocess runner is monkeypatched, so no test touches the network,
spawns a real test runner, or sleeps on the wall clock for ordering.  Every
asynchronous test sets an explicit ``max_cycles`` (or requests a stop) because
``max_cycles=0`` deliberately means "run forever".
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeAlias

import pytest

# ``TesterAgent`` is aliased so pytest does not try to collect it as a test class.
from furrow.agents.tester import CheckResult
from furrow.agents.tester import TesterAgent as _TesterAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

GOAL = "Add a health endpoint to the service"

#: Wall-clock guard so a regression that hangs the loop fails instead of hanging.
RUN_TIMEOUT = 15.0

#: Distinct model names, so the fake client can tell the roles apart from the
#: ``model`` argument alone instead of depending on the wording of the prompts.
FAKE_PLANNER_MODEL = "fake-planner-model"
FAKE_WORKER_MODEL = "fake-worker-model"
FAKE_TESTER_MODEL = "fake-tester-model"

PASSING_CHECK = CheckResult(
    name="pytest",
    command="pytest -q",
    returncode=0,
    output="3 passed in 0.02s",
)
FAILING_CHECK = CheckResult(
    name="pytest",
    command="pytest -q",
    returncode=1,
    output="tests/test_health.py::test_health FAILED\nassert 200 == 404",
)
FAILURE_TEXT = "tests/test_health.py::test_health FAILED"

#: Signature of the per-task worker hook the fake client installs.
WorkerHook: TypeAlias = Callable[[str], Awaitable[str]]


class Recorder:
    """Collects the events the orchestrator emits."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.events.append(event)

    @property
    def kinds(self) -> list[str]:
        return [str(event["kind"]) for event in self.events]

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["kind"] == kind]

    def lifecycle(self) -> list[tuple[str, Any]]:
        return [
            (str(event["kind"]), event.get("task_id"))
            for event in self.events
            if event["kind"] in ("task_start", "task_end")
        ]


class FakeClient(LLMClient):
    """Duck-typed stand-in for ``LLMClient`` that never calls a provider."""

    def __init__(
        self,
        settings: Settings,
        *,
        plans: list[Plan],
        tests: list[TestResult] | None = None,
        worker: WorkerHook | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.plans = list(plans)
        self.tests = list(tests or [TestResult(passed=True, summary="all green", failures=[])])
        self.worker = worker
        self.plan_prompts: list[str] = []
        self.worker_prompts: list[str] = []
        self.tester_prompts: list[str] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        role = _role(prompt, model, self.settings)
        if role == "planner":
            self.plan_prompts.append(prompt)
            index = min(len(self.plan_prompts) - 1, len(self.plans) - 1)
            return self.plans[index].model_dump_json()
        if role == "tester":
            self.tester_prompts.append(prompt)
            index = min(len(self.tester_prompts) - 1, len(self.tests) - 1)
            return self.tests[index].model_dump_json()
        self.worker_prompts.append(prompt)
        if self.worker is not None:
            return await self.worker(prompt)
        return "changed one file"


def _role(prompt: str, model: str | None, settings: Settings) -> str:
    """Which agent a completion belongs to: planner, tester, or worker.

    The requested model is authoritative and unambiguous because the fake
    settings use a distinct name per role; the prompt wording is only a
    fallback.
    """
    for role, name in (
        ("planner", settings.planner_model),
        ("tester", settings.tester_model),
        ("worker", settings.worker_model),
    ):
        if model is not None and model == name:
            return role
    lowered = prompt.lower()
    if "planning agent" in lowered:
        return "planner"
    if "tester agent" in lowered:
        return "tester"
    return "worker"


def _settings(**overrides: Any) -> Settings:
    """Baseline settings with an explicit cycle budget, cap, and role models."""
    base: dict[str, Any] = {
        "max_cycles": 3,
        "max_parallel_tasks": 5,
        "planner_model": FAKE_PLANNER_MODEL,
        "worker_model": FAKE_WORKER_MODEL,
        "tester_model": FAKE_TESTER_MODEL,
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)


def _task(
    task_id: str, description: str = "do the work", dependencies: list[str] | None = None
) -> TaskModel:
    return TaskModel(id=task_id, description=description, dependencies=dependencies or [])


def _plan(*tasks: TaskModel, rationale: str = "small slice") -> Plan:
    return Plan(tasks=list(tasks), rationale=rationale)


def _failing_result(summary: str = "1 failed") -> TestResult:
    return TestResult(passed=False, summary=summary, failures=[FAILURE_TEXT])


@pytest.fixture
def checks(monkeypatch: pytest.MonkeyPatch) -> Callable[[CheckResult], None]:
    """Replace the tester subprocess runner with a canned check outcome."""

    def install(result: CheckResult) -> None:
        async def fake_run_checks(self: _TesterAgent) -> list[CheckResult]:
            return [result]

        monkeypatch.setattr(_TesterAgent, "_run_checks", fake_run_checks)

    return install


async def _run(orchestrator: Orchestrator) -> dict[str, Any]:
    return await asyncio.wait_for(orchestrator.run(), timeout=RUN_TIMEOUT)


# 1. Regression: the loop used to halt after one cycle because _get_tasks()
#    returned [], making ``0 >= 0`` true.


async def test_more_than_one_cycle_runs_while_tests_keep_failing(checks: Any) -> None:
    checks(FAILING_CHECK)
    client = FakeClient(
        _settings(max_cycles=3), plans=[_plan(_task("1"))], tests=[_failing_result()]
    )
    recorder = Recorder()
    orchestrator = Orchestrator(goal=GOAL, client=client, on_event=recorder)

    state = await _run(orchestrator)

    assert state["cycles"] == 3
    assert len(recorder.of("cycle_start")) == 3
    assert len(recorder.of("cycle_end")) == 3
    assert len(client.plan_prompts) == 3
    assert state["stopped_reason"] == "cycle_budget_exhausted"


# 2. max_cycles is honored exactly, and 0 means "run until something stops it".


@pytest.mark.parametrize("budget", [1, 2, 4])
async def test_max_cycles_is_respected_exactly(budget: int, checks: Any) -> None:
    checks(FAILING_CHECK)
    client = FakeClient(
        _settings(max_cycles=budget),
        plans=[_plan(_task("1"))],
        tests=[_failing_result("still red")],
    )
    recorder = Recorder()
    orchestrator = Orchestrator(goal=GOAL, client=client, on_event=recorder)

    state = await _run(orchestrator)

    assert state["cycles"] == budget
    assert state["max_cycles"] == budget
    run_end = recorder.of("run_end")[0]
    assert run_end["cycles"] == budget
    assert run_end["reason"] == "cycle_budget_exhausted"
    assert state["stopped_reason"] == "cycle_budget_exhausted"
    assert len(recorder.of("cycle_end")) == budget


async def test_budget_of_one_cycle_still_returns_state(checks: Any) -> None:
    checks(PASSING_CHECK)
    client = FakeClient(_settings(max_cycles=1), plans=[_plan(_task("1"))])

    state = await _run(Orchestrator(goal=GOAL, client=client))

    assert state["cycles"] == 1
    assert state["completed"] == 1
    assert state["failed"] == 0
    assert state["stopped_reason"] == "goal_complete"


async def test_zero_max_cycles_runs_until_a_stop_request(checks: Any) -> None:
    """``max_cycles=0`` is unlimited: only the stop request ends the loop."""
    checks(FAILING_CHECK)
    client = FakeClient(
        _settings(max_cycles=0, max_parallel_tasks=2),
        plans=[_plan(_task("1"))],
        tests=[_failing_result()],
    )
    recorder = Recorder()
    orchestrator: Orchestrator

    def on_event(event: dict[str, Any]) -> None:
        recorder(event)
        if event["kind"] == "cycle_start" and event["cycle"] >= 3:
            orchestrator.request_stop()

    orchestrator = Orchestrator(goal=GOAL, client=client, on_event=on_event)
    state = await _run(orchestrator)

    assert len(recorder.of("cycle_start")) == 3
    assert state["cycles"] == 3
    assert state["max_cycles"] == 0
    assert state["stopped_reason"] == "stop_requested"


# 3. request_stop() ends the loop from inside the callback and from a thread.


async def test_request_stop_from_the_event_callback_ends_the_run(checks: Any) -> None:
    checks(PASSING_CHECK)
    client = FakeClient(_settings(max_cycles=5), plans=[_plan(_task("1"), _task("2"))])
    recorder = Recorder()
    orchestrator: Orchestrator

    def on_event(event: dict[str, Any]) -> None:
        recorder(event)
        # Requested once cycle one finished cleanly, so the stop is the only
        # thing that can end the run.
        if event["kind"] == "cycle_end":
            orchestrator.request_stop()

    orchestrator = Orchestrator(goal=GOAL, client=client, on_event=on_event)
    state = await _run(orchestrator)

    assert state["cycles"] == 1
    assert state["stopped_reason"] == "stop_requested"
    assert recorder.kinds.count("stop_requested") == 1
    assert recorder.kinds[0] == "run_start"
    assert recorder.kinds[-1] == "run_end"
    assert recorder.kinds.count("cycle_end") == 1
    # The first cycle really did run to completion before the stop landed.
    assert len(recorder.of("plan")) == 1
    assert len(client.worker_prompts) == 2
    assert recorder.of("test_result")[0]["passed"] is True


async def test_request_stop_from_another_thread_ends_the_run(checks: Any) -> None:
    checks(PASSING_CHECK)
    started = asyncio.Event()
    release = asyncio.Event()

    async def worker(prompt: str) -> str:
        started.set()
        await release.wait()
        return "done"

    client = FakeClient(
        # A cap of one keeps the sibling deterministically parked on the
        # semaphore, so the stop check inside the guard is what ends it.
        _settings(max_cycles=4, max_parallel_tasks=1),
        plans=[_plan(_task("1"), _task("2"))],
        worker=worker,
    )
    orchestrator = Orchestrator(goal=GOAL, client=client)
    run_task = asyncio.create_task(orchestrator.run())

    await asyncio.wait_for(started.wait(), timeout=RUN_TIMEOUT)
    thread = threading.Thread(target=orchestrator.request_stop)
    thread.start()
    thread.join(timeout=RUN_TIMEOUT)
    release.set()

    state = await asyncio.wait_for(run_task, timeout=RUN_TIMEOUT)
    assert state["stopped_reason"] == "stop_requested"
    assert state["cycles"] == 1
    assert state["running"] is False
    # The sibling task in the same level never started: it was waiting on the
    # semaphore, and the stop was honoured before it reached the worker.
    assert len(client.worker_prompts) == 1


# 4. The goal is never clobbered by failure text.


async def test_original_goal_survives_failing_cycles(checks: Any) -> None:
    checks(FAILING_CHECK)
    client = FakeClient(
        _settings(max_cycles=2), plans=[_plan(_task("1"))], tests=[_failing_result()]
    )
    orchestrator = Orchestrator(goal=GOAL, client=client)

    state = await _run(orchestrator)

    assert state["original_goal"] == GOAL
    assert orchestrator.original_goal == GOAL
    # The planning prompt carries the original goal plus the failure context.
    assert GOAL in state["goal"]
    assert state["goal"] != GOAL
    assert FAILURE_TEXT in state["goal"]
    assert GOAL in client.plan_prompts[0]
    assert FAILURE_TEXT not in client.plan_prompts[0]
    assert GOAL in client.plan_prompts[1]
    assert FAILURE_TEXT in client.plan_prompts[1]
    # The tester judges the work against the original goal.
    assert GOAL in client.tester_prompts[0]
    assert state["feedback"] is not None


# 5. Dependency-ordered scheduling.


async def test_dependent_task_starts_only_after_its_dependency(checks: Any) -> None:
    checks(PASSING_CHECK)
    plan = _plan(
        _task("1", "build the base"),
        _task("2", "consume the base", dependencies=["1"]),
        _task("3", "consume 2 as well", dependencies=["2"]),
    )
    client = FakeClient(_settings(max_cycles=1), plans=[plan])
    recorder = Recorder()

    state = await _run(Orchestrator(goal=GOAL, client=client, on_event=recorder))

    assert recorder.lifecycle() == [
        ("task_start", "1"),
        ("task_end", "1"),
        ("task_start", "2"),
        ("task_end", "2"),
        ("task_start", "3"),
        ("task_end", "3"),
    ]
    assert state["completed"] == 3
    assert state["failed"] == 0


async def test_unknown_dependency_is_reported_instead_of_hanging(checks: Any) -> None:
    checks(PASSING_CHECK)
    plan = _plan(_task("1"), _task("2", dependencies=["99"]))
    client = FakeClient(_settings(max_cycles=1), plans=[plan])

    state = await _run(Orchestrator(goal=GOAL, client=client))

    assert state["failed"] == 2
    assert client.worker_prompts == []
    assert any("unknown task id" in (task["result"] or "") for task in state["tasks"])
    assert any("unknown task id" in error and "99" in error for error in state["errors"])
    assert state["stopped_reason"] == "cycle_budget_exhausted"


async def test_dependency_cycle_is_reported_instead_of_hanging(checks: Any) -> None:
    checks(PASSING_CHECK)
    plan = _plan(_task("1", dependencies=["2"]), _task("2", dependencies=["1"]))
    client = FakeClient(_settings(max_cycles=1), plans=[plan])

    state = await _run(Orchestrator(goal=GOAL, client=client))

    assert state["failed"] == 2
    assert client.worker_prompts == []
    assert any("dependency cycle" in (task["result"] or "") for task in state["tasks"])
    assert any("dependency cycle" in error for error in state["errors"])
    assert state["stopped_reason"] == "cycle_budget_exhausted"


async def test_duplicate_task_ids_are_reported(checks: Any) -> None:
    checks(PASSING_CHECK)
    plan = _plan(_task("1"), _task("1"))
    client = FakeClient(_settings(max_cycles=1), plans=[plan])

    state = await _run(Orchestrator(goal=GOAL, client=client))

    assert state["failed"] == 2
    assert client.worker_prompts == []
    assert any("duplicate task id" in (task["result"] or "") for task in state["tasks"])
    assert any("duplicate task id" in error for error in state["errors"])


# 6. Concurrency cap.


async def test_concurrency_cap_of_one_serializes_independent_tasks(checks: Any) -> None:
    checks(PASSING_CHECK)
    in_flight = 0
    peak = 0

    async def worker(prompt: str) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        for _ in range(5):
            await asyncio.sleep(0)  # yield, never wall-clock waiting
        in_flight -= 1
        return "done"

    client = FakeClient(
        _settings(max_cycles=1, max_parallel_tasks=1),
        plans=[_plan(_task("1"), _task("2"), _task("3"))],
        worker=worker,
    )

    state = await _run(Orchestrator(goal=GOAL, client=client))

    assert peak == 1
    assert state["max_parallel_tasks"] == 1
    assert state["completed"] == 3
    assert len(client.worker_prompts) == 3


async def test_default_cap_lets_independent_tasks_overlap(checks: Any) -> None:
    checks(PASSING_CHECK)
    in_flight = 0
    peak = 0
    everyone_started = asyncio.Event()

    async def worker(prompt: str) -> str:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        if in_flight >= 3:
            # Deterministic rendezvous: nothing proceeds until all three are up.
            everyone_started.set()
        await everyone_started.wait()
        in_flight -= 1
        return "done"

    client = FakeClient(
        _settings(max_cycles=1, max_parallel_tasks=5),
        plans=[_plan(_task("1"), _task("2"), _task("3"))],
        worker=worker,
    )

    state = await _run(Orchestrator(goal=GOAL, client=client))

    assert peak == 3
    assert state["max_parallel_tasks"] == 5
    assert state["completed"] == 3


async def test_dependency_levels_bound_the_cap_too(checks: Any) -> None:
    """Independent tasks may overlap, but dependents never start early."""
    checks(PASSING_CHECK)
    running: list[str] = []
    overlap = 0

    async def worker(prompt: str) -> str:
        nonlocal overlap
        running.append(prompt)
        overlap = max(overlap, len(running))
        for _ in range(3):
            await asyncio.sleep(0)
        running.pop()
        return "done"

    plan = _plan(
        _task("1", "first half"),
        _task("2", "other half"),
        _task("3", "join them", dependencies=["1", "2"]),
    )
    client = FakeClient(_settings(max_cycles=1, max_parallel_tasks=4), plans=[plan], worker=worker)

    state = await _run(Orchestrator(goal=GOAL, client=client))

    assert overlap == 2
    assert state["completed"] == 3


# 7. A raising worker fails only its own task.


async def test_worker_exception_fails_one_task_and_the_run_continues(checks: Any) -> None:
    checks(PASSING_CHECK)

    async def worker(prompt: str) -> str:
        if "explode" in prompt:
            raise RuntimeError("kaboom")
        return "fine"

    plan = _plan(_task("1", "ordinary work"), _task("2", "please explode here"))
    client = FakeClient(_settings(max_cycles=2), plans=[plan], worker=worker)
    recorder = Recorder()

    state = await _run(Orchestrator(goal=GOAL, client=client, on_event=recorder))

    by_id = {task["id"]: task for task in state["tasks"]}
    assert by_id["1"]["status"] == "completed"
    assert by_id["2"]["status"] == "failed"
    assert "kaboom" in by_id["2"]["result"]
    assert "RuntimeError" in by_id["2"]["result"]
    assert state["completed"] == 1
    assert state["failed"] == 1
    # The cycle still finished: the tester ran, and the loop kept going.
    assert len(client.tester_prompts) == 2
    assert state["cycles"] == 2
    assert len(recorder.of("run_end")) == 1


async def test_base_exception_from_a_worker_is_coerced_to_text(checks: Any) -> None:
    """A ``BaseException`` is recorded as text, never stored in the ``str`` field."""

    class Boom(BaseException):
        """Not an ``Exception``: the loop still has to survive it."""

    checks(PASSING_CHECK)

    async def worker(prompt: str) -> str:
        if "detonate" in prompt:
            raise Boom("detonated")
        return "fine"

    plan = _plan(_task("1", "ordinary work"), _task("2", "detonate right here"))
    client = FakeClient(_settings(max_cycles=1), plans=[plan], worker=worker)

    state = await _run(Orchestrator(goal=GOAL, client=client))

    by_id = {task["id"]: task for task in state["tasks"]}
    assert by_id["2"]["status"] == "failed"
    assert isinstance(by_id["2"]["result"], str)
    assert "Boom" in by_id["2"]["result"]
    assert "detonated" in by_id["2"]["result"]
    assert state["completed"] == 1
    assert state["failed"] == 1
    assert state["stopped_reason"] == "cycle_budget_exhausted"


# 8. Event contract.


async def test_event_sequence_for_a_clean_single_cycle_run(checks: Any) -> None:
    checks(PASSING_CHECK)
    client = FakeClient(
        _settings(max_cycles=1), plans=[_plan(_task("1"), _task("2"), rationale="because")]
    )
    recorder = Recorder()

    state = await _run(Orchestrator(goal=GOAL, client=client, on_event=recorder))

    assert recorder.kinds == [
        "run_start",
        "cycle_start",
        "plan",
        "task_start",
        "task_end",
        "task_start",
        "task_end",
        "test_result",
        "cycle_end",
        "run_end",
    ]
    for event in recorder.events:
        assert isinstance(event["kind"], str)
        assert isinstance(event["cycle"], int)
    assert recorder.of("run_start")[0]["goal"] == GOAL
    plan_event = recorder.of("plan")[0]
    assert plan_event["rationale"] == "because"
    assert [task["id"] for task in plan_event["tasks"]] == ["1", "2"]
    assert plan_event["tasks"][0]["status"] == "pending"
    test_event = recorder.of("test_result")[0]
    assert test_event["passed"] is True
    assert test_event["failures"] == []
    assert recorder.of("task_end")[0]["status"] == "completed"
    run_end = recorder.of("run_end")[0]
    assert run_end["reason"] == "goal_complete"
    assert run_end["cycles"] == 1
    assert run_end["completed"] == 2
    assert run_end["failed"] == 0
    assert state["stopped_reason"] == "goal_complete"


async def test_no_callback_means_no_events(checks: Any) -> None:
    checks(PASSING_CHECK)
    client = FakeClient(_settings(max_cycles=2), plans=[_plan(_task("1"))])
    orchestrator = Orchestrator(goal=GOAL, client=client)

    state = await _run(orchestrator)

    assert orchestrator._emit_errors == 0
    assert state["cycles"] == 1
    assert state["completed"] == 1


async def test_raising_callback_never_breaks_the_run(checks: Any) -> None:
    checks(FAILING_CHECK)
    client = FakeClient(
        _settings(max_cycles=3),
        plans=[_plan(_task("1"))],
        tests=[_failing_result("red")],
    )
    calls = 0

    def explode(event: dict[str, Any]) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("listener is broken")

    orchestrator = Orchestrator(goal=GOAL, client=client, on_event=explode)
    state = await _run(orchestrator)

    assert calls > 0
    assert state["cycles"] == 3
    assert state["stopped_reason"] == "cycle_budget_exhausted"
    assert orchestrator._emit_errors == calls


# 9. State snapshot.


async def test_state_is_json_serializable(checks: Any) -> None:
    checks(FAILING_CHECK)
    client = FakeClient(
        _settings(max_cycles=2),
        plans=[_plan(_task("1"))],
        tests=[_failing_result("red")],
    )
    orchestrator = Orchestrator(goal=GOAL, client=client)

    state = await _run(orchestrator)
    decoded = json.loads(json.dumps(state))

    for key in (
        "goal",
        "original_goal",
        "cycles",
        "running",
        "stopped_reason",
        "tasks",
        "completed",
        "failed",
        "unfinished",
        "max_cycles",
        "max_parallel_tasks",
        "tests_passed",
        "feedback",
        "last_test_result",
        "errors",
    ):
        assert key in state
    assert decoded["original_goal"] == GOAL
    assert decoded["tasks"][0]["status"] == "completed"
    assert decoded["last_test_result"]["passed"] is False
    # The live snapshot and the returned one agree.
    assert orchestrator.state == state


async def test_is_running_flips_around_the_run(checks: Any) -> None:
    checks(PASSING_CHECK)
    observed: list[bool] = []
    orchestrator: Orchestrator

    async def worker(prompt: str) -> str:
        observed.append(orchestrator.is_running)
        return "done"

    client = FakeClient(_settings(max_cycles=1), plans=[_plan(_task("1"))], worker=worker)
    orchestrator = Orchestrator(goal=GOAL, client=client)

    assert orchestrator.is_running is False
    state = await _run(orchestrator)

    assert observed == [True]
    assert state["running"] is False
    assert orchestrator.is_running is False


async def test_plan_with_no_tasks_reports_no_tasks_planned(checks: Any) -> None:
    checks(PASSING_CHECK)
    client = FakeClient(_settings(max_cycles=5), plans=[_plan()])
    recorder = Recorder()

    state = await _run(Orchestrator(goal=GOAL, client=client, on_event=recorder))

    assert state["cycles"] == 1
    assert state["stopped_reason"] == "no_tasks_planned"
    assert state["tasks"] == []
    assert recorder.kinds[-1] == "run_end"