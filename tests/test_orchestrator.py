"""Tests for :mod:`furrow.core.orchestrator`.

The planner, workers and tester are replaced with recording doubles, so the whole loop runs
without an LLM, a subprocess or a wall clock. The headline regression covered here is the old
``_get_tasks()`` stub that made ``_is_done()`` unconditionally true, which stopped the
"infinite" loop after a single cycle.

Every run is wrapped in a deadline (``bounded``) so a scheduler deadlock fails loudly instead of
hanging the suite, and every test that needs coordination uses :class:`asyncio.Event` rather than
a sleep.
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from furrow.config import Plan, Settings, TaskModel, TaskStatus
from furrow.core.orchestrator import DEFAULT_MAX_PARALLEL, STALL_LIMIT, LoopResult, Orchestrator
from tests.conftest import (
    AgentStubs,
    EventRecorder,
    FakeLLMClient,
    failing,
    make_plan,
    passing,
)

pytestmark = pytest.mark.usefixtures("captured_console")

ALL_EVENT_NAMES = [
    "run.start",
    "cycle.start",
    "plan",
    "task.start",
    "task.done",
    "task.failed",
    "task.skipped",
    "test.result",
    "cycle.done",
    "run.done",
]

StubInstaller = Callable[..., AgentStubs]


def make_orchestrator(
    client: FakeLLMClient,
    *,
    goal: str = "ship the feature",
    on_event: Callable[[str, dict[str, Any]], Any] | None = None,
) -> Orchestrator:
    return Orchestrator(goal=goal, client=client, on_event=on_event)


async def run_with_deadline(
    orchestrator: Orchestrator,
    bounded: Callable[..., Awaitable[LoopResult]],
    timeout: float = 5.0,
) -> LoopResult:
    """Run the loop under a deadline and fail loudly if the deadline fired.

    ``Orchestrator.run`` swallows ``CancelledError`` and returns ``stopped_reason="cancelled"``
    (its documented contract), so a deadlock shows up as a cancelled result rather than as a
    ``TimeoutError`` from ``asyncio.wait_for``. Checking the reason turns that into a clear
    failure instead of a hung suite.
    """
    result = await bounded(orchestrator.run(), timeout)
    assert result.stopped_reason != "cancelled", "the loop never finished; it hit the deadline"
    return result


# --------------------------------------------------------------------------------------
# The loop actually loops
# --------------------------------------------------------------------------------------


async def test_is_done_stops_after_first_cycle_when_tests_pass(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stubs = stub_agents(plans=[make_plan("do it")], test_results=[passing()])
    orchestrator = make_orchestrator(make_fake_llm(), on_event=event_recorder)

    result = await run_with_deadline(orchestrator, bounded)

    assert result.cycles == 1
    assert result.stopped_reason == "goal-complete"
    assert result.passed is True
    assert result.tasks_completed == 1
    assert stubs.planner.calls == 1


async def test_loop_runs_another_cycle_while_tests_keep_failing(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stubs = stub_agents(
        plans=[make_plan("try again"), make_plan("try again"), make_plan("try again")],
        test_results=[failing("boom"), failing("boom"), passing()],
    )
    orchestrator = make_orchestrator(make_fake_llm())

    result = await run_with_deadline(orchestrator, bounded)

    assert result.cycles == 3, "the loop must not exit after the first failing cycle"
    assert result.stopped_reason == "goal-complete"
    assert result.passed is True
    assert stubs.planner.calls == 3


async def test_a_failing_task_keeps_the_loop_alone_going_even_when_tests_pass(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    # _is_done() requires passing tests *and* zero failed tasks.
    stub_agents(
        plans=[make_plan("attempt"), make_plan("attempt")],
        test_results=[passing(), passing()],
        worker_results={"1": RuntimeError("could not write the file")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=2))

    result = await run_with_deadline(orchestrator, bounded)

    assert result.cycles == 2, "a failed task means the goal is not done yet"
    assert result.passed is False, "a failed task means the goal was not achieved"
    assert result.tasks_failed == 2


async def test_a_task_that_always_fails_while_tests_pass_stalls_the_loop(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    """Regression: a forever-failing task must halt the loop, not spin it forever.

    ``_is_done()`` refuses to finish while a task failed, so the stall detector is the only
    thing that can stop this run. The failing task is keyed on its description plus its error
    text (never on ``TaskModel.id``, which the planner re-mints every cycle), so the
    fingerprint repeats and the loop halts after ``STALL_LIMIT`` cycles on its own -- with the
    default ``max_cycles=0`` ("unlimited") the deadline below is only a safety net.
    """
    stubs = stub_agents(
        plans=[make_plan("attempt")],
        test_results=[passing()],
        worker_results={"1": RuntimeError("could not write the file")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=0), on_event=event_recorder)

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "stalled", "the loop must stop on its own, not on the deadline"
    assert result.cycles == STALL_LIMIT
    assert result.passed is False
    assert stubs.planner.calls == STALL_LIMIT, "the loop must not keep planning after the stall"
    assert any("tasks keep failing" in failure for failure in result.failures)
    assert any("could not write the file" in failure for failure in result.failures)
    done = event_recorder.payloads("run.done")[-1]
    assert "tasks keep failing" in done["message"]


async def test_a_task_that_fails_twice_then_succeeds_never_stalls_the_loop(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    """No false positive: an intermittent failure is progress, not a stuck state."""
    attempts: list[str] = []

    async def flaky(task: TaskModel) -> None:
        attempts.append(task.id)
        if len(attempts) <= 2:
            raise RuntimeError("flaky worker")

    stub_agents(
        plans=[make_plan("flaky task")],
        test_results=[passing(), passing(), passing()],
        worker_hook=flaky,
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=0))

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "goal-complete"
    assert result.cycles == 3, "two failing cycles below the stall limit are tolerated"
    assert result.passed is True
    assert len(attempts) == 3


async def test_two_different_failing_tasks_are_not_one_stuck_state(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    """No false positive: two tasks that fail identically but do different work are not "stuck".

    Both tasks raise the *same* error text and differ only in id and description, so a
    fingerprint keyed on the error alone (or on nothing at all) would look stuck after three
    cycles. Alternating descriptions mean the fingerprint genuinely differs each cycle, so the
    run keeps going and finishes on the seventh.
    """

    def plan_for(task_id: str, description: str) -> Plan:
        return Plan(
            tasks=[TaskModel(id=task_id, description=description)],
            rationale="alternate",
        )

    stub_agents(
        plans=[
            plan_for("a", "alpha"),
            plan_for("b", "beta"),
            plan_for("a", "alpha"),
            plan_for("b", "beta"),
            plan_for("a", "alpha"),
            plan_for("b", "beta"),
            plan_for("c", "gamma"),
        ],
        test_results=[passing()],
        worker_results={"a": RuntimeError("boom"), "b": RuntimeError("boom")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=0))

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "goal-complete", "alternating work is not a stall"
    assert result.cycles == 7
    assert result.passed is True
    assert result.tasks_completed == 1
    assert result.tasks_failed == 6


async def test_failure_text_drift_does_not_defeat_the_stall_detector(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    """Whitespace and case drift is cosmetic, so it must still count as the same stuck state."""
    forms = [
        "could not write the file",
        "  Could NOT   write\n the file ",
        "COULD not Write the  file",
    ]
    attempts: list[str] = []

    async def drifting_worker(task: TaskModel) -> None:
        attempts.append(task.id)
        raise RuntimeError(forms[len(attempts) % len(forms)])

    stub_agents(
        plans=[make_plan("write the file")],
        test_results=[passing()],
        worker_hook=drifting_worker,
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=0))

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "stalled"
    assert result.cycles == STALL_LIMIT
    assert len(attempts) == STALL_LIMIT
    stall = next(failure for failure in result.failures if "stalled" in failure)
    assert "failed task write the file" in stall
    assert "could not write the file" in stall


async def test_a_tester_that_raises_every_cycle_stalls_instead_of_spinning(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    """A cycle with no usable test result still has to count toward a stall."""
    stub_agents(plans=[make_plan("x")], test_results=[RuntimeError("tester exploded")])
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=0))

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "stalled"
    assert result.cycles == STALL_LIMIT
    stall = next(failure for failure in result.failures if "stalled" in failure)
    assert "identical test failures" in stall
    assert "tester exploded" in stall


async def test_task_statuses_are_enum_members_so_serialization_stays_quiet(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    """Regression: assigning raw strings made ``plan.model_dump()`` warn on every cycle."""
    stub_agents(
        plans=[make_plan("done", "boom")],
        test_results=[passing()],
        worker_results={"2": RuntimeError("nope")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=1))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "max-cycles"
    serializer_noise = [
        str(entry.message) for entry in caught if "serializer" in str(entry.message).lower()
    ]
    assert serializer_noise == [], f"pydantic serializer warnings: {serializer_noise}"
    assert [type(task.status) for task in orchestrator.tasks] == [TaskStatus, TaskStatus]
    assert orchestrator.tasks[0].status is TaskStatus.COMPLETED
    assert orchestrator.tasks[1].status is TaskStatus.FAILED


# --------------------------------------------------------------------------------------
# Termination reasons
# --------------------------------------------------------------------------------------


async def test_empty_plan_stops_with_no_tasks(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[Plan(tasks=[], rationale="nothing left")], test_results=[])
    orchestrator = make_orchestrator(make_fake_llm())

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "no-tasks"
    assert result.passed is False, "an empty plan is not a silent success"
    assert result.cycles == 1


async def test_max_cycles_setting_stops_the_loop(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("again")], test_results=[failing("boom")])
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=2))

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "max-cycles"
    assert result.cycles == 2
    assert result.passed is False


async def test_max_cycles_of_zero_means_unlimited_but_the_stall_check_still_stops(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("again")], test_results=[failing("same failure")])
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=0))

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "stalled"
    assert result.cycles == STALL_LIMIT


async def test_identical_failures_stall_the_loop_instead_of_spinning(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("again")], test_results=[failing("same failure")])
    orchestrator = make_orchestrator(make_fake_llm())

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "stalled"
    assert result.cycles == STALL_LIMIT
    assert result.passed is False
    assert any("stalled" in failure for failure in result.failures)
    assert any("same failure" in failure for failure in result.failures)


async def test_changing_failures_do_not_count_as_a_stall(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(
        plans=[make_plan("again")],
        test_results=[failing(f"failure {index}") for index in range(4)] + [passing()],
    )
    orchestrator = make_orchestrator(make_fake_llm())

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "goal-complete"
    assert result.cycles == 5


async def test_two_identical_failures_are_below_the_stall_limit(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(
        plans=[make_plan("again")],
        test_results=[failing("same"), failing("same"), passing()],
    )
    orchestrator = make_orchestrator(make_fake_llm())

    result = await run_with_deadline(orchestrator, bounded)

    assert result.cycles == 3, "STALL_LIMIT identical failures are tolerated"
    assert result.stopped_reason == "goal-complete"


async def test_planner_value_error_degrades_to_a_stall_instead_of_crashing(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stubs = stub_agents(plans=[ValueError("model returned prose")], test_results=[])
    orchestrator = make_orchestrator(make_fake_llm(), on_event=event_recorder)

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "stalled"
    assert result.cycles == STALL_LIMIT
    assert stubs.worker.calls == [], "a planner failure must not run any worker"
    assert event_recorder.count("test.result") == 0
    assert any("Planner failed" in message for message in event_recorder.messages("cycle.done"))


async def test_a_planner_recovers_after_a_transient_failure(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(
        plans=[ValueError("bad json"), make_plan("retry worked")],
        test_results=[passing()],
    )
    orchestrator = make_orchestrator(make_fake_llm())

    result = await run_with_deadline(orchestrator, bounded)

    assert result.cycles == 2
    assert result.stopped_reason == "goal-complete"


async def test_a_crashed_cycle_is_counted_and_can_stall_the_loop(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("boom")], test_results=[RuntimeError("tester exploded")])
    orchestrator = make_orchestrator(make_fake_llm(), on_event=event_recorder)

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "max-cycles" or result.stopped_reason == "stalled"
    assert event_recorder.count("cycle.done") >= result.cycles


async def test_a_tester_exception_becomes_a_failing_test_result(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(
        plans=[make_plan("x"), make_plan("x")],
        test_results=[RuntimeError("tester exploded"), passing()],
    )
    orchestrator = make_orchestrator(make_fake_llm(), on_event=event_recorder)

    result = await run_with_deadline(orchestrator, bounded)

    payloads = event_recorder.payloads("test.result")
    assert payloads[0]["passed"] is False
    assert "tester exploded" in payloads[0]["summary"]
    assert result.cycles == 2


# --------------------------------------------------------------------------------------
# Goal and focus
# --------------------------------------------------------------------------------------


async def test_the_original_goal_is_preserved_across_a_failure_cycle(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stubs = stub_agents(
        plans=[make_plan("fix it"), make_plan("fix it")],
        test_results=[failing("tests/test_auth.py::test_login"), passing()],
    )
    orchestrator = make_orchestrator(make_fake_llm(), goal="implement jwt auth")

    await run_with_deadline(orchestrator, bounded)

    assert orchestrator.goal == "implement jwt auth"
    first_prompt, second_prompt = stubs.planner.goals
    assert first_prompt == "implement jwt auth"
    assert second_prompt.startswith("implement jwt auth")
    assert "Current focus:" in second_prompt
    assert "tests/test_auth.py::test_login" in second_prompt


async def test_focus_is_cleared_once_the_cycle_passes(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(
        plans=[make_plan("x")],
        test_results=[passing()],
    )
    orchestrator = make_orchestrator(make_fake_llm())

    await run_with_deadline(orchestrator, bounded)

    assert orchestrator.focus is None


async def test_a_failed_task_is_added_to_the_focus_instruction(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stubs = stub_agents(
        plans=[make_plan("x"), make_plan("x")],
        test_results=[passing(), passing()],
        worker_results={"1": RuntimeError("worker could not apply the edit")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=2))

    await run_with_deadline(orchestrator, bounded)

    assert stubs.planner.goals[1].count("Current focus:") == 1
    assert "worker could not apply the edit" in stubs.planner.goals[1]


# --------------------------------------------------------------------------------------
# Scheduler: dependencies, ordering and concurrency
# --------------------------------------------------------------------------------------


async def test_a_task_whose_dependency_failed_is_skipped_and_never_runs(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="a", description="first"),
            TaskModel(id="b", description="second", dependencies=["a"]),
        ],
        rationale="ordered",
    )
    stubs = stub_agents(
        plans=[plan],
        test_results=[passing()],
        worker_results={"a": RuntimeError("a exploded")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=1), on_event=event_recorder)

    result = await run_with_deadline(orchestrator, bounded)

    assert stubs.worker.started_ids == ["a"], "b must never run"
    assert "b" not in [task.id for task in orchestrator.tasks if task.status == "completed"]
    skipped = orchestrator.tasks[1]
    assert skipped.status == "skipped"
    assert skipped.result == "Dependency failed or was skipped"
    payload = event_recorder.payloads("task.skipped")[0]
    assert payload["task_id"] == "b"
    assert result.tasks_failed == 1
    assert result.passed is False


async def test_a_dependency_chain_runs_in_declaration_order(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    order: list[str] = []

    async def record(task: TaskModel) -> None:
        order.append(task.id)

    plan = Plan(
        tasks=[
            TaskModel(id="c", description="third", dependencies=["b"]),
            TaskModel(id="b", description="second", dependencies=["a"]),
            TaskModel(id="a", description="first"),
        ],
        rationale="chained",
    )
    stub_agents(plans=[plan], test_results=[passing()], worker_hook=record)
    orchestrator = make_orchestrator(make_fake_llm())

    await run_with_deadline(orchestrator, bounded)

    assert order == ["a", "b", "c"]


async def test_independent_tasks_run_concurrently(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    started = 0
    inflight = 0
    peak = 0
    all_running = asyncio.Event()
    release = asyncio.Event()

    async def hold(task: TaskModel) -> None:
        nonlocal started, inflight, peak
        started += 1
        inflight += 1
        peak = max(peak, inflight)
        if started == 3:
            all_running.set()
        await release.wait()
        inflight -= 1

    plan = Plan(
        tasks=[TaskModel(id=str(index), description="independent") for index in range(1, 4)],
        rationale="parallel",
    )
    stub_agents(plans=[plan], test_results=[passing()], worker_hook=hold)
    orchestrator = make_orchestrator(make_fake_llm(max_parallel_tasks=3))

    run = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(all_running.wait(), timeout=5)
    assert peak == 3, "three independent tasks must be in flight at the same time"
    release.set()

    result = await bounded(run)
    assert result.stopped_reason != "cancelled", "the parallel run hit its deadline"
    assert result.tasks_completed == 3


async def test_max_parallel_tasks_caps_in_flight_workers(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    inflight = 0
    peak = 0
    release = asyncio.Event()
    two_running = asyncio.Event()

    async def hold(task: TaskModel) -> None:
        nonlocal inflight, peak
        inflight += 1
        peak = max(peak, inflight)
        if inflight == 2:
            two_running.set()
        await release.wait()
        inflight -= 1

    plan = Plan(
        tasks=[TaskModel(id=str(index), description="capped") for index in range(1, 6)],
        rationale="five tasks",
    )
    stub_agents(plans=[plan], test_results=[passing()], worker_hook=hold)
    orchestrator = make_orchestrator(make_fake_llm(max_parallel_tasks=2))

    run = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(two_running.wait(), timeout=5)
    assert inflight == 2
    release.set()

    result = await bounded(run)
    assert result.stopped_reason != "cancelled", "the capped run hit its deadline"
    assert peak == 2, f"the semaphore must cap in-flight workers, peak was {peak}"
    assert result.tasks_completed == 5


async def test_dependency_cycle_is_failed_instead_of_hanging(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="a", description="needs b", dependencies=["b"]),
            TaskModel(id="b", description="needs a", dependencies=["a"]),
        ],
        rationale="circular",
    )
    stubs = stub_agents(plans=[plan], test_results=[passing()])
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=1), on_event=event_recorder)

    result = await run_with_deadline(orchestrator, bounded)

    assert stubs.worker.calls == [], "a cycle must not start any worker"
    assert result.tasks_failed == 2
    for task in orchestrator.tasks:
        assert task.status == "failed"
        assert "Unresolvable dependencies" in str(task.result)
    assert event_recorder.count("task.failed") == 2


async def test_unknown_dependency_id_is_failed_instead_of_hanging(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    plan = Plan(
        tasks=[TaskModel(id="a", description="needs ghost", dependencies=["ghost"])],
        rationale="dangling",
    )
    stubs = stub_agents(plans=[plan], test_results=[passing()])
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=1))

    result = await run_with_deadline(orchestrator, bounded)

    assert stubs.worker.calls == []
    assert result.tasks_failed == 1


async def test_a_worker_exception_marks_the_task_failed_and_emits_an_event(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stubs = stub_agents(
        plans=[make_plan("x"), make_plan("x")],
        test_results=[passing(), passing()],
        worker_results={"1": ValueError("nope")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=2), on_event=event_recorder)

    await run_with_deadline(orchestrator, bounded)

    payload = event_recorder.payloads("task.failed")[0]
    assert payload["task_id"] == "1"
    assert "nope" in payload["error"]
    assert stubs.tester.calls == 2, "a failed task still runs the tests"


# --------------------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------------------


async def test_every_documented_event_is_emitted_with_a_message(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="ok", description="finishes"),
            TaskModel(id="bad", description="explodes"),
            TaskModel(id="blocked", description="waits for bad", dependencies=["bad"]),
        ],
        rationale="mixed",
    )
    stub_agents(
        plans=[plan],
        test_results=[passing()],
        worker_results={"bad": RuntimeError("kaboom")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=1), on_event=event_recorder)

    await run_with_deadline(orchestrator, bounded)

    emitted = set(event_recorder.names)
    assert set(ALL_EVENT_NAMES) <= emitted, f"missing: {sorted(set(ALL_EVENT_NAMES) - emitted)}"
    for name, payload in event_recorder.events:
        assert "message" in payload, f"{name} payload has no message: {payload}"
        assert isinstance(payload["message"], str) and payload["message"]


async def test_run_start_and_run_done_carry_the_goal_and_the_reason(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    orchestrator = make_orchestrator(
        make_fake_llm(), goal="a specific goal", on_event=event_recorder
    )

    await run_with_deadline(orchestrator, bounded)

    start = event_recorder.payloads("run.start")[0]
    assert start["goal"] == "a specific goal"
    assert event_recorder.names[0] == "run.start"
    assert event_recorder.names[-1] == "run.done"
    done = event_recorder.payloads("run.done")[0]
    assert done["reason"] == "goal-complete"
    assert done["cycles"] == 1


async def test_plan_event_lists_the_planned_task_ids(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("one", "two", "three")], test_results=[passing()])
    orchestrator = make_orchestrator(make_fake_llm(), on_event=event_recorder)

    await run_with_deadline(orchestrator, bounded)

    payload = event_recorder.payloads("plan")[0]
    assert payload["tasks"] == ["1", "2", "3"]
    assert payload["cycle"] == 1
    assert payload["rationale"] == "because"


async def test_test_result_event_reports_the_verdict_and_failures(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("x"), make_plan("x")], test_results=[failing("boom"), passing()])
    orchestrator = make_orchestrator(make_fake_llm(), on_event=event_recorder)

    await run_with_deadline(orchestrator, bounded)

    first, second = event_recorder.payloads("test.result")
    assert first["passed"] is False
    assert first["failures"] == ["boom"]
    assert second["passed"] is True


async def test_task_done_event_carries_the_task_id(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    event_recorder: EventRecorder,
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    orchestrator = make_orchestrator(make_fake_llm(), on_event=event_recorder)

    await run_with_deadline(orchestrator, bounded)

    payload = event_recorder.payloads("task.done")[0]
    assert payload["task_id"] == "1"
    assert payload["status"] == "completed"


async def test_a_raising_event_handler_does_not_kill_the_loop(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("x"), make_plan("x")], test_results=[failing("boom"), passing()])
    seen: list[str] = []

    def explode(name: str, payload: dict[str, Any]) -> None:
        seen.append(name)
        raise RuntimeError("the consumer died")

    orchestrator = make_orchestrator(make_fake_llm(), on_event=explode)

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "goal-complete"
    assert result.cycles == 2
    assert "run.start" in seen and "run.done" in seen


async def test_an_async_event_handler_is_awaited(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    finished: list[str] = []

    async def handler(name: str, payload: dict[str, Any]) -> None:
        await asyncio.sleep(0)
        finished.append(name)

    orchestrator = make_orchestrator(make_fake_llm(), on_event=handler)

    await run_with_deadline(orchestrator, bounded)

    assert finished == event_names_for_single_passing_cycle()
    assert finished[-1] == "run.done"


def event_names_for_single_passing_cycle() -> list[str]:
    return [
        "run.start",
        "cycle.start",
        "plan",
        "task.start",
        "task.done",
        "test.result",
        "cycle.done",
        "run.done",
    ]


async def test_a_returned_awaitable_from_a_sync_handler_is_awaited(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    finished: list[str] = []

    async def record(name: str, payload: dict[str, Any]) -> None:
        finished.append(name)

    def handler(name: str, payload: dict[str, Any]) -> Awaitable[None]:
        return record(name, payload)

    orchestrator = make_orchestrator(make_fake_llm(), on_event=handler)

    await run_with_deadline(orchestrator, bounded)

    assert "run.done" in finished


# --------------------------------------------------------------------------------------
# Cancellation
# --------------------------------------------------------------------------------------


async def test_cancellation_returns_a_loop_result_instead_of_propagating(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    planning = asyncio.Event()

    async def never_finishes(task: TaskModel) -> None:
        planning.set()
        await asyncio.Event().wait()

    stub_agents(
        plans=[make_plan("x")],
        test_results=[passing()],
        worker_hook=never_finishes,
    )
    orchestrator = make_orchestrator(make_fake_llm())

    run = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(planning.wait(), timeout=5)
    run.cancel()

    result = await bounded(run)
    assert isinstance(result, LoopResult)
    assert result.stopped_reason == "cancelled"
    assert result.passed is False


async def test_cancellation_marks_unfinished_tasks_as_cancelled(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    planning = asyncio.Event()

    async def never_finishes(task: TaskModel) -> None:
        planning.set()
        await asyncio.Event().wait()

    stub_agents(plans=[make_plan("x")], test_results=[passing()], worker_hook=never_finishes)
    orchestrator = make_orchestrator(make_fake_llm())

    run = asyncio.create_task(orchestrator.run())
    await asyncio.wait_for(planning.wait(), timeout=5)
    run.cancel()
    await bounded(run)

    assert [task.status for task in orchestrator.tasks] == ["cancelled"]


async def test_cancellation_before_the_first_cycle_is_still_a_result(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    orchestrator = make_orchestrator(make_fake_llm())

    run = asyncio.create_task(orchestrator.run())
    await asyncio.sleep(0)
    run.cancel()

    result = await bounded(run)
    assert result.stopped_reason == "cancelled"
    assert result.cycles in (0, 1)


# --------------------------------------------------------------------------------------
# LoopResult and settings plumbing
# --------------------------------------------------------------------------------------


def test_loop_result_optional_fields_default_to_empty() -> None:
    result = LoopResult(goal="g", cycles=1, stopped_reason="goal-complete", passed=True)
    assert result.tasks_completed == 0
    assert result.tasks_failed == 0
    assert result.failures == []


def test_loop_result_failures_are_not_shared_between_instances() -> None:
    first = LoopResult(goal="g", cycles=1, stopped_reason="stalled", passed=False)
    second = LoopResult(goal="g", cycles=1, stopped_reason="stalled", passed=False)
    first.failures.append("boom")
    assert second.failures == []


async def test_loop_result_failures_carry_test_failures_and_failed_tasks(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(
        plans=[make_plan("x")],
        test_results=[failing("test output noise")],
        worker_results={"1": RuntimeError("worker noise")},
    )
    orchestrator = make_orchestrator(make_fake_llm(max_cycles=1))

    result = await run_with_deadline(orchestrator, bounded)

    assert "test output noise" in result.failures
    assert any("worker noise" in failure for failure in result.failures)


def test_parallel_limit_falls_back_to_the_default_for_a_nonsense_setting(
    settings: Settings,
) -> None:
    orchestrator = Orchestrator(goal="g", client=FakeLLMClient(settings))
    assert orchestrator._max_parallel() == DEFAULT_MAX_PARALLEL

    broken = settings.model_copy(update={"max_parallel_tasks": 0})
    limited = Orchestrator(goal="g", client=FakeLLMClient(broken))
    assert limited._max_parallel() == DEFAULT_MAX_PARALLEL


def test_max_cycles_of_a_nonsense_setting_is_treated_as_unlimited(settings: Settings) -> None:
    negative = settings.model_copy(update={"max_cycles": -3})
    assert Orchestrator(goal="g", client=FakeLLMClient(negative))._max_cycles() == 0


async def test_orchestrator_without_an_event_handler_runs_quietly(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    orchestrator = make_orchestrator(make_fake_llm())

    result = await run_with_deadline(orchestrator, bounded)

    assert result.stopped_reason == "goal-complete"


async def test_the_loop_uses_a_single_client_for_every_agent(
    stub_agents: StubInstaller,
    make_fake_llm: Callable[..., FakeLLMClient],
    bounded: Callable[..., Awaitable[Any]],
) -> None:
    client = make_fake_llm()
    stub_agents(plans=[make_plan("x")], test_results=[passing()])
    orchestrator = Orchestrator(goal="g", client=client)

    await run_with_deadline(orchestrator, bounded)

    assert orchestrator.client is client
