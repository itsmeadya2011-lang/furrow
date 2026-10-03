"""End-to-end orchestrator tests: real agents, canned LLM replies, no network, no keys."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from furrow.agents import tester as tester_module
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

GOAL = "add a health check endpoint"
PLANNER_MODEL = "stub-planner"
WORKER_MODEL = "stub-worker"
# Every orchestrator test is bounded by default; the "unlimited" (max_cycles == 0)
# cases must ask for it explicitly and supply their own deterministic terminator.
MAX_CYCLES = 3

# Imported for assertions only; pytest would otherwise try to collect TestResult.
TestResult.__test__ = False


def make_settings(**overrides: Any) -> Settings:
    """Settings with per-role model names so the fake client can dispatch by model."""
    base: dict[str, Any] = {
        "planner_model": PLANNER_MODEL,
        "worker_model": WORKER_MODEL,
        "max_cycles": MAX_CYCLES,
        "log_level": "CRITICAL",  # keep structlog out of the captured test output
    }
    base.update(overrides)
    return Settings(**base)


def plan_json(
    *tasks: tuple[str, str] | tuple[str, str, list[str]], rationale: str = "stub plan"
) -> str:
    """Canned planner reply; a task tuple is (id, description) or (id, description, deps)."""
    payload = [
        {"id": task[0], "description": task[1], "dependencies": list(task[2])} if len(task) == 3
        else {"id": task[0], "description": task[1], "dependencies": []}
        for task in tasks
    ]
    return json.dumps({"tasks": payload, "rationale": rationale})


def plan_object(
    *tasks: tuple[str, str] | tuple[str, str, list[str]], rationale: str = "stub plan"
) -> Plan:
    """The same plan as :func:`plan_json`, as a validated ``Plan``."""
    return Plan.model_validate(json.loads(plan_json(*tasks, rationale=rationale)))


def stub_planner(monkeypatch: pytest.MonkeyPatch, *plans: Plan | BaseException) -> None:
    """Hand the orchestrator canned plans, bypassing the real planner's normalization.

    ``PlannerAgent`` clamps a plan to ``max(1, min(5, max_parallel_tasks))`` tasks and
    deletes back-edges that close a dependency cycle, so a plan that goes through the
    real planner can never arrive cyclic or wider than the cap. The orchestrator keeps
    its own defences against both shapes; testing them requires supplying those shapes
    directly. Replies are consumed in order and the last one repeats, matching
    :class:`FakeClient`; a queued exception is raised like a planner blow-up.
    """
    queue = list(plans)

    async def fake_plan(self: Any, goal: str, test_failures: Any = None) -> Plan:
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        return item

    monkeypatch.setattr(orchestrator_module.PlannerAgent, "plan", fake_plan)


def verdict(
    passed: bool, summary: str = "stub summary", failures: tuple[str, ...] = ()
) -> TestResult:
    """Canned tester verdict replayed by the stubbed ``TesterAgent.run``.

    Deliberately not named ``test_*``: pytest would collect and run it as a
    test case (it takes non-fixture arguments and errors at collection).
    """
    return TestResult(passed=passed, summary=summary, failures=list(failures))


class FakeClient(LLMClient):
    """Replays canned replies keyed by model. Touching a real provider is a test bug."""

    def __init__(
        self,
        settings: Settings,
        plans: list[Any],
        test_results: list[Any] | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self.plans = list(plans)
        # Replayed by the stubbed TesterAgent.run, not by complete(): these
        # tests exercise loop behaviour, not the tester's LLM plumbing.
        self.test_results = list(test_results or [])
        self.plan_calls = 0
        self.worker_calls = 0
        self.prompts: list[tuple[str, str]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.prompts.append((model or "", prompt))
        if model == PLANNER_MODEL:
            return self._reply(self.plans, "plan_calls", "planner")
        if model == WORKER_MODEL:
            self.worker_calls += 1
            # Yield control so real overlap (and the parallel cap) is observable.
            await asyncio.sleep(0.01)
            return f"implemented {self.worker_calls} task(s)"
        # The tester is stubbed at TesterAgent.run, so a request for the
        # tester model here would mean the orchestrator bypassed the agent.
        raise AssertionError(f"unexpected model requested: {model!r}")

    def _reply(self, queue: list[Any], counter: str, who: str) -> str:
        if not queue:
            raise AssertionError(f"{who} was called but no canned reply was queued")
        index = getattr(self, counter)
        setattr(self, counter, index + 1)
        item = queue[min(index, len(queue) - 1)]  # the last reply repeats forever
        if isinstance(item, BaseException):
            raise item
        return item

    def planner_prompts(self) -> list[str]:
        return [prompt for model, prompt in self.prompts if model == PLANNER_MODEL]

    @property
    def anthropic(self) -> Any:
        raise AssertionError("FakeClient must not build a real Anthropic client")

    @property
    def openai(self) -> Any:
        raise AssertionError("FakeClient must not build a real OpenAI client")


class WorkerTrace:
    """Records worker start/end order, peak concurrency and the agents themselves."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []
        self.instances: list[WorkerAgent] = []
        self.in_flight = 0
        self.max_in_flight = 0

    @property
    def started(self) -> list[str]:
        return [task_id for kind, task_id in self.events if kind == "start"]


def trace_workers(monkeypatch: pytest.MonkeyPatch, fail_ids: tuple[str, ...] = ()) -> WorkerTrace:
    """Wrap WorkerAgent.run to observe scheduling; optionally make chosen tasks crash."""
    trace = WorkerTrace()
    original = WorkerAgent.run

    async def traced(self: WorkerAgent) -> str:
        trace.events.append(("start", self.task.id))
        trace.instances.append(self)
        trace.in_flight += 1
        trace.max_in_flight = max(trace.max_in_flight, trace.in_flight)
        try:
            if self.task.id in fail_ids:
                raise RuntimeError(f"task {self.task.id} exploded")
            return await original(self)
        finally:
            trace.in_flight -= 1
            trace.events.append(("end", self.task.id))

    monkeypatch.setattr(WorkerAgent, "run", traced)
    return trace


@pytest.fixture(autouse=True)
def stub_tester_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the seam the orchestrator actually calls: ``TesterAgent.run``.

    The stub replays the canned verdicts queued on the client (FakeClient
    keeps them in ``test_results``), so these tests never depend on the
    tester's internals - runner detection, evidence collection, LLM judging -
    and keep working through tester refactors. A client without a queue gets
    a clean pass, which is what most single-cycle tests need.
    """

    async def fake_run(self: Any, goal: str, tasks: list[TaskModel]) -> TestResult:
        queue = getattr(self.client, "test_results", None)
        if not queue:
            return TestResult(passed=True, summary="stub tester: pass")
        # The last verdict repeats forever, mirroring FakeClient's plan replies.
        item = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, TestResult):
            return item
        return TestResult.model_validate(json.loads(item))

    # Patch through the orchestrator module's reference: that is the exact
    # class object the loop instantiates, and keeping TesterAgent out of this
    # module's namespace stops pytest from trying to collect it as a test.
    monkeypatch.setattr(orchestrator_module.TesterAgent, "run", fake_run)


def run_orchestrator(
    goal: str, client: FakeClient, events: list[str] | None = None
) -> Orchestrator:
    orchestrator = Orchestrator(
        goal=goal,
        client=client,
        on_event=events.append if events is not None else None,
        settings=client.settings,
    )
    asyncio.run(orchestrator.run())
    return orchestrator


# --------------------------------------------------------------------- tests


def test_loop_keeps_running_while_tests_fail() -> None:
    """Regression: _get_tasks() used to return [], so the loop exited after cycle 1."""
    client = FakeClient(
        make_settings(max_cycles=5),
        plans=[plan_json(("1", "first slice")), plan_json(("2", "second slice"))],
        test_results=[
            verdict(False, "1 failure", ("test_health FAILED",)),
            verdict(True, "All tests passed"),
        ],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert orchestrator.cycles == 2
    assert orchestrator.stop_reason == "goal complete"
    assert orchestrator.is_done is True
    assert [task.id for task in orchestrator.tasks] == ["2"]
    assert orchestrator.tasks[0].status == "completed"
    assert any(line.startswith("Cycle 1: tests failed (1 failure)") for line in events)
    assert "Cycle 2: plan ready with 1 task(s)" in events
    assert "Cycle 2: task 2 completed" in events
    assert "Cycle 2: tests passed (All tests passed)" in events
    assert orchestrator.tasks[0].result == "implemented 2 task(s)"


def test_stops_with_goal_complete_when_tasks_and_tests_pass() -> None:
    client = FakeClient(
        make_settings(),
        plans=[plan_json(("1", "a"), ("2", "b"))],
        test_results=[verdict(True, "All tests passed")],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert orchestrator.cycles == 1
    assert orchestrator.is_done is True
    assert orchestrator.stop_reason == "goal complete"
    assert [task.status for task in orchestrator.tasks] == ["completed", "completed"]
    assert any(line.startswith("Goal complete:") for line in events)


def test_empty_plan_stops_instead_of_spinning(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plan with no tasks is not proof of success: stop and say exactly that.

    The real planner raises on an empty task list rather than returning one, so the
    empty plan is produced at the planner seam; what is under test here is the
    orchestrator's own guard against "nothing to do" masquerading as "goal met".
    """
    stub_planner(monkeypatch, Plan(tasks=[], rationale="nothing to do"))
    client = FakeClient(
        make_settings(max_cycles=0),  # unlimited: only the empty plan can stop this
        plans=[],
        test_results=[],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert orchestrator.cycles == 1
    assert orchestrator.stop_reason == "planner returned no tasks"
    assert orchestrator.is_done is False
    assert "Cycle 1: planner returned no tasks" in events
    assert "Stopping: planner returned no tasks" in events


def test_max_cycles_is_respected() -> None:
    client = FakeClient(
        make_settings(max_cycles=2),
        plans=[plan_json(("1", "a"))],
        test_results=[verdict(False, "1 failure", ("same failure",))],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert orchestrator.cycles == 2
    assert orchestrator.stop_reason == "max cycles reached (2)"
    assert orchestrator.is_done is False
    assert "Stopping: max cycles reached (2)" in events


def test_identical_failures_stop_an_unlimited_loop() -> None:
    client = FakeClient(
        make_settings(max_cycles=0),  # 0 == unlimited, so the progress guard must fire
        plans=[plan_json(("1", "a"))],
        test_results=[verdict(False, "3 failures", ("E one", "E two", "E three"))],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert orchestrator.cycles == 3
    assert orchestrator.stop_reason == "no progress after 3 identical failures"
    assert orchestrator.is_done is False
    assert "Stopping: no progress after 3 identical failures" in events


def test_reordered_failures_count_as_no_progress() -> None:
    client = FakeClient(
        make_settings(max_cycles=0),
        plans=[plan_json(("1", "a"))],
        test_results=[
            verdict(False, "2 failures", ("E one", "E two")),
            verdict(False, "2 failures", ("E two", "E one")),
            verdict(False, "2 failures", ("E one", "E two")),
        ],
    )
    orchestrator = run_orchestrator(GOAL, client)

    assert orchestrator.stop_reason == "no progress after 3 identical failures"


def test_changing_failures_keep_the_loop_alive() -> None:
    client = FakeClient(
        make_settings(max_cycles=4),
        plans=[plan_json(("1", "a"))],
        test_results=[
            verdict(False, "1 failure", ("E one",)),
            verdict(False, "1 failure", ("E two",)),
            verdict(False, "1 failure", ("E three",)),
            verdict(True, "All tests passed"),
        ],
    )
    orchestrator = run_orchestrator(GOAL, client)

    assert orchestrator.cycles == 4
    assert orchestrator.stop_reason == "goal complete"


def test_planner_exception_is_caught_and_the_loop_survives() -> None:
    client = FakeClient(
        make_settings(max_cycles=5),
        plans=[RuntimeError("planner exploded"), plan_json(("1", "a"))],
        test_results=[verdict(True, "All tests passed")],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert orchestrator.cycles == 2
    assert orchestrator.stop_reason == "goal complete"
    # The underlying cause reaches the event stream, so a user can see why a cycle was lost.
    assert any("planner failed (planner exploded)" in line for line in events)
    assert orchestrator.is_done is True


def test_three_consecutive_planner_failures_stop_the_loop() -> None:
    client = FakeClient(
        make_settings(max_cycles=0),
        plans=[ValueError("Failed to parse plan from LLM")],
        test_results=[],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert orchestrator.cycles == 3
    assert orchestrator.stop_reason == "planner failed 3 cycles in a row"
    assert orchestrator.is_done is False
    # The terminal "Stopping:" line also names the reason, so count only the
    # per-cycle failure events: exactly three consecutive planner blow-ups.
    cycle_events = [line for line in events if line.startswith("Cycle")]
    assert sum(1 for line in cycle_events if "planner failed" in line) == 3


def test_tester_exception_does_not_crash_the_loop() -> None:
    client = FakeClient(
        make_settings(max_cycles=5),
        plans=[plan_json(("1", "a"))],
        test_results=[RuntimeError("tester exploded"), verdict(True, "All tests passed")],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    # A tester blow-up is recorded as a failing cycle (visible in the events), and the
    # loop carries on to a real verdict instead of dying with it.
    assert "Cycle 1: tests failed (1 failure)" in events
    assert orchestrator.cycles == 2
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is True
    assert orchestrator.stop_reason == "goal complete"


def test_failed_dependency_skips_dependents(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = trace_workers(monkeypatch, fail_ids=("1",))
    client = FakeClient(
        make_settings(max_cycles=1),
        plans=[plan_json(("1", "base"), ("2", "dependent", ["1"]), ("3", "grandchild", ["2"]))],
        test_results=[verdict(True, "All tests passed")],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    statuses = {task.id: task.status for task in orchestrator.tasks}
    assert statuses == {"1": "failed", "2": "skipped", "3": "skipped"}
    # Dependents must never be handed to a worker, and the failure must be recorded.
    assert trace.started == ["1"]
    assert orchestrator.tasks[0].result is not None
    assert "RuntimeError" in orchestrator.tasks[0].result
    assert "Cycle 1: task 1 failed" in events
    assert "Cycle 1: task 2 skipped (dependency did not complete)" in events
    # Tests were green but work is unfinished, so the goal is not met.
    assert orchestrator.is_done is False
    assert orchestrator.stop_reason == "max cycles reached (1)"


def test_dependency_waves_run_after_their_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = trace_workers(monkeypatch)
    client = FakeClient(
        make_settings(),
        plans=[plan_json(("1", "a"), ("2", "b", ["1"]), ("3", "c", ["1"]))],
        test_results=[verdict(True, "All tests passed")],
    )
    orchestrator = run_orchestrator(GOAL, client)

    assert trace.started == ["1", "2", "3"]
    assert trace.events.index(("end", "1")) < trace.events.index(("start", "2"))
    assert trace.events.index(("end", "1")) < trace.events.index(("start", "3"))
    # Independent tasks in one wave still overlap.
    assert trace.max_in_flight == 2
    assert [task.status for task in orchestrator.tasks] == ["completed"] * 3
    assert orchestrator.is_done is True


def test_dependency_cycles_are_skipped_not_deadlocked(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cyclic plan cannot be unlocked by any wave, so its members are skipped.

    The plan is injected at the planner seam because the real planner deletes
    back-edges that close a cycle; the orchestrator's own cycle detection is the
    defence for a plan that arrives cyclic from anywhere.
    """
    stub_planner(monkeypatch, plan_object(("1", "a", ["2"]), ("2", "b", ["1"])))
    trace = trace_workers(monkeypatch)
    client = FakeClient(
        make_settings(max_cycles=1),
        plans=[],
        test_results=[verdict(True, "All tests passed")],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert [task.status for task in orchestrator.tasks] == ["skipped", "skipped"]
    assert trace.started == []
    assert "Cycle 1: task 1 skipped (circular dependency)" in events
    assert orchestrator.is_done is False


def test_parallel_cap_limits_concurrency(monkeypatch: pytest.MonkeyPatch) -> None:
    """The semaphore caps a wave; the planner seam is stubbed so the plan is wider than the cap."""
    stub_planner(monkeypatch, plan_object(*[(str(n), f"task {n}") for n in range(1, 6)]))
    trace = trace_workers(monkeypatch)
    client = FakeClient(
        make_settings(max_parallel_tasks=2),
        plans=[],
        test_results=[verdict(True, "All tests passed")],
    )
    orchestrator = run_orchestrator(GOAL, client)

    assert trace.started == ["1", "2", "3", "4", "5"]  # the cap delays, never reorders
    assert trace.max_in_flight == 2
    assert [task.status for task in orchestrator.tasks] == ["completed"] * 5


def test_zero_parallel_cap_means_unbounded(monkeypatch: pytest.MonkeyPatch) -> None:
    """``max_parallel_tasks == 0`` is "no cap": a Semaphore(0) would deadlock the wave."""
    stub_planner(monkeypatch, plan_object(("1", "a"), ("2", "b"), ("3", "c")))
    trace = trace_workers(monkeypatch)
    client = FakeClient(
        make_settings(max_parallel_tasks=0),
        plans=[],
        test_results=[verdict(True, "All tests passed")],
    )
    run_orchestrator(GOAL, client)

    assert trace.max_in_flight == 3


def test_failing_worker_does_not_abort_its_siblings(monkeypatch: pytest.MonkeyPatch) -> None:
    trace = trace_workers(monkeypatch, fail_ids=("2",))
    client = FakeClient(
        make_settings(max_cycles=1),
        plans=[plan_json(("1", "a"), ("2", "b"), ("3", "c"))],
        test_results=[verdict(True, "All tests passed")],
    )
    orchestrator = run_orchestrator(GOAL, client)

    assert sorted(trace.started) == ["1", "2", "3"]
    assert {task.id: task.status for task in orchestrator.tasks} == {
        "1": "completed",
        "2": "failed",
        "3": "completed",
    }


def test_worker_sees_dependency_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """A dependent task is handed what its dependency produced."""
    trace = trace_workers(monkeypatch)
    client = FakeClient(
        make_settings(),
        plans=[plan_json(("1", "base"), ("2", "dependent", ["1"]))],
        test_results=[verdict(True, "All tests passed")],
    )
    orchestrator = run_orchestrator(GOAL, client)

    by_id = {worker.task.id: worker for worker in trace.instances}
    assert by_id["1"].dependency_results == {}
    assert by_id["2"].dependency_results == {"1": "implemented 1 task(s)"}
    assert orchestrator.is_done is True


def test_worker_without_dependency_results_kwarg(monkeypatch: pytest.MonkeyPatch) -> None:
    """The dependency_results kwarg is probed, not assumed: no kwarg, no crash.

    A worker built before the parameter existed still runs; had the
    orchestrator passed the kwarg anyway, this constructor would have raised
    TypeError and both tasks would have failed instead of completing.
    """

    class LegacyWorker:
        def __init__(self, task: TaskModel, client: Any = None) -> None:
            self.task = task

        async def run(self) -> str:
            return f"legacy {self.task.id}"

    monkeypatch.setattr(orchestrator_module, "WorkerAgent", LegacyWorker)
    client = FakeClient(
        make_settings(),
        plans=[plan_json(("1", "a"), ("2", "b", ["1"]))],
        test_results=[verdict(True, "All tests passed")],
    )
    orchestrator = run_orchestrator(GOAL, client)

    assert [task.id for task in orchestrator.tasks] == ["1", "2"]
    assert [task.status for task in orchestrator.tasks] == ["completed", "completed"]
    assert orchestrator.is_done is True


def test_no_runner_vacuous_pass_is_a_pass_not_a_failure() -> None:
    """A project without a test runner is vacuously verified: done, not stuck.

    The old loop read "No test runner found." as a failing test and steered
    the goal into "fix the tests" forever; a vacuous pass must end the run.
    """
    client = FakeClient(
        make_settings(),
        plans=[plan_json(("1", "a"))],
        test_results=[verdict(True, summary=tester_module.NO_RUNNER_SUMMARY)],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)

    assert orchestrator.cycles == 1
    assert orchestrator.is_done is True
    assert orchestrator.stop_reason == "goal complete"
    # No failures means no failure steering: the focus stays on the goal.
    assert orchestrator.focus == GOAL


def test_goal_is_never_clobbered_and_planner_sees_goal_and_focus() -> None:
    client = FakeClient(
        make_settings(max_cycles=5),
        plans=[plan_json(("1", "a")), plan_json(("2", "b"))],
        test_results=[
            verdict(False, "1 failure", ("test_health FAILED",)),
            verdict(True, "All tests passed"),
        ],
    )
    orchestrator = run_orchestrator(GOAL, client)

    assert orchestrator.goal == GOAL
    assert "test_health FAILED" in orchestrator.focus
    prompts = client.planner_prompts()
    assert len(prompts) == 2
    assert GOAL in prompts[0]
    assert GOAL in prompts[1]
    assert "test_health FAILED" in prompts[1]


def test_events_are_single_line_plain_text() -> None:
    client = FakeClient(
        make_settings(),
        plans=[plan_json(("1", "a"))],
        test_results=[verdict(False, "multi\nline\nsummary", ("boom\nin the middle",))],
    )
    events: list[str] = []
    run_orchestrator(GOAL, client, events)

    assert events
    for line in events:
        assert "\n" not in line
        assert line == line.strip()
        # No Rich markup: a stray "[" here would be swallowed or mis-rendered downstream.
        assert "[" not in line
        assert "]" not in line
    assert any(line.startswith("Cycle 1: tests failed (") for line in events)


def test_raising_on_event_does_not_break_the_loop() -> None:
    events: list[str] = []

    def exploding_callback(line: str) -> None:
        events.append(line)
        raise RuntimeError("consumer is broken")

    client = FakeClient(
        make_settings(),
        plans=[plan_json(("1", "a"))],
        test_results=[verdict(True, "All tests passed")],
    )
    orchestrator = Orchestrator(
        goal=GOAL, client=client, on_event=exploding_callback, settings=client.settings
    )
    asyncio.run(orchestrator.run())

    assert events
    assert orchestrator.cycles == 1
    assert orchestrator.stop_reason == "goal complete"
    assert orchestrator.is_done is True


def test_is_done_is_false_before_a_tester_run_and_while_cycles_fail() -> None:
    client = FakeClient(
        make_settings(max_cycles=1),
        plans=[plan_json(("1", "a"))],
        test_results=[verdict(False, "1 failure", ("boom",))],
    )
    events: list[str] = []
    orchestrator = Orchestrator(
        goal=GOAL, client=client, on_event=events.append, settings=client.settings
    )

    # Nothing has been planned or verified yet: an unstarted run is not a finished one.
    assert orchestrator.is_done is False
    assert orchestrator.tasks == []
    assert orchestrator.stop_reason == ""

    asyncio.run(orchestrator.run())

    assert orchestrator.cycles == 1
    assert orchestrator.is_done is False
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is False
    # All tasks finished, yet the goal is not met: the tests are the authority.
    assert all(task.status == "completed" for task in orchestrator.tasks)


def test_cosmetic_churn_stops_after_max_stagnant_cycles() -> None:
    """Old exact-fingerprint guard would never have tripped on varying timings."""
    client = FakeClient(
        make_settings(max_cycles=0),
        plans=[plan_json(("1", "a"))],
        test_results=[
            verdict(False, "same", ("test_auth.py::test_login expected 3 got 4 in 0.13s",)),
            verdict(False, "same", ("test_auth.py::test_login expected 3 got 4 in 0.21s",)),
            verdict(False, "same", ("test_auth.py::test_login expected 3 got 4 in 0.35s",)),
        ],
    )
    orchestrator = run_orchestrator(GOAL, client)
    assert orchestrator.cycles == 3
    assert orchestrator.stop_reason == "no progress after 3 identical failures"


def test_genuinely_improving_sequence_does_not_trigger_stagnation() -> None:
    """Each cycle removes a failure, so the loop must keep running."""
    client = FakeClient(
        make_settings(max_cycles=0, max_stagnant_cycles=3, stagnation_threshold=0.8),
        plans=[plan_json(("1", "a"))],
        test_results=[
            verdict(False, "2 failures", ("E one", "E two")),
            verdict(False, "1 failure", ("E one",)),
            verdict(False, "1 failure", ("E two",)),
            verdict(True, "All tests passed"),
        ],
    )
    orchestrator = run_orchestrator(GOAL, client)
    assert orchestrator.cycles == 4
    assert orchestrator.stop_reason == "goal complete"


def test_disjoint_failures_do_not_accumulate_stagnation() -> None:
    """Completely different failures every cycle must never trip the guard."""
    client = FakeClient(
        make_settings(max_cycles=0, max_stagnant_cycles=3, stagnation_threshold=0.8),
        plans=[plan_json(("1", "a"))],
        test_results=[
            verdict(False, "1 failure", ("E one",)),
            verdict(False, "1 failure", ("E two",)),
            verdict(False, "1 failure", ("E three",)),
            verdict(True, "All tests passed"),
        ],
    )
    orchestrator = run_orchestrator(GOAL, client)
    assert orchestrator.cycles == 4
    assert orchestrator.stop_reason == "goal complete"


def test_stagnation_counter_resets_after_progress_cycle() -> None:
    """A cycle with a different failure set resets the counter."""
    client = FakeClient(
        make_settings(max_cycles=0, max_stagnant_cycles=3, stagnation_threshold=0.8),
        plans=[plan_json(("1", "a"))],
        test_results=[
            verdict(False, "1 failure", ("E one",)),
            verdict(False, "1 failure", ("E one",)),
            verdict(False, "1 failure", ("E two",)),
            verdict(False, "1 failure", ("E one",)),
            verdict(False, "1 failure", ("E one",)),
            verdict(False, "1 failure", ("E one",)),
        ],
    )
    orchestrator = run_orchestrator(GOAL, client)
    assert orchestrator.cycles == 6
    assert orchestrator.stop_reason == "no progress after 3 identical failures"


def test_stagnation_settings_overrides_are_honoured() -> None:
    """Lowering the threshold or max cycles should change the behaviour."""
    client = FakeClient(
        make_settings(max_cycles=0, max_stagnant_cycles=2, stagnation_threshold=0.5),
        plans=[plan_json(("1", "a"))],
        test_results=[
            verdict(False, "2 failures", ("E one", "E two")),
            verdict(False, "2 failures", ("E two", "E one")),
        ],
    )
    orchestrator = run_orchestrator(GOAL, client)
    assert orchestrator.cycles == 2
    assert orchestrator.stop_reason == "no progress after 2 identical failures"


def test_stagnation_stop_reason_and_event_mention_progress() -> None:
    """The terminal event must say stagnation, not success."""
    client = FakeClient(
        make_settings(max_cycles=0),
        plans=[plan_json(("1", "a"))],
        test_results=[verdict(False, "1 failure", ("E one",))],
    )
    events: list[str] = []
    orchestrator = run_orchestrator(GOAL, client, events)
    assert orchestrator.stop_reason == "no progress after 3 identical failures"
    assert "no progress" in orchestrator.stop_reason.lower()
    assert "Stopping: no progress after 3 identical failures" in events


def test_nonsense_stagnation_settings_do_not_crash() -> None:
    """Out-of-range values fall back to defaults and the loop still terminates."""
    client = FakeClient(
        make_settings(max_cycles=0, max_stagnant_cycles=0, stagnation_threshold=2.0),
        plans=[plan_json(("1", "a"))],
        test_results=[verdict(False, "1 failure", ("E one",))],
    )
    orchestrator = run_orchestrator(GOAL, client)
    assert orchestrator.cycles == 3
    assert orchestrator.stop_reason == "no progress after 3 identical failures"
