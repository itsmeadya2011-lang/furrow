"""Regression tests for :class:`furrow.core.orchestrator.Orchestrator`.

This is where the hardening pass changed the most, so these tests pin the loop itself:

* the run keeps cycling instead of stopping after one pass;
* "done" means *every task completed* **and** the tester said the suite passes;
* ``max_cycles`` and ``max_parallel_tasks`` bound the run and the fan-out;
* dependencies gate execution and blocked tasks are reported, not silently dropped;
* the original goal is never rewritten by a failure and is always re-sent to the planner;
* progress events are plain text and carry a terminal reason;
* untrusted goals, task ids and test output cannot kill the loop through rich markup;
* cancellation propagates and never marks a task completed.

Every async test here is a *sync* function that drives an inner coroutine with
``asyncio.run`` -- ``pytest-asyncio`` is not installed, so ``asyncio_mode = "auto"`` is
inert and a bare ``async def test_*`` would not run at all.
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Callable

import pytest

from furrow.config import Plan, Settings, TaskModel
from furrow.config import settings as singleton_settings
from furrow.core.orchestrator import Orchestrator

from .conftest import FAILING_VERDICT, PASSING_VERDICT, StubLLMClient, plan_json, task_json

ONE_TASK = plan_json(task_json("a", "Do A"))

#: Every rich tag the orchestrator authors. None of these may reach an ``on_event`` consumer.
RICH_TAGS = ("[bold", "[red", "[green", "[yellow", "[cyan", "[italic", "[dim", "[blue")

#: Fingerprint of the defect the failure-handling tests below were written to pin. The failure
#: branches used to hand the caught exception *object* to ``rich.markup.escape()``, which needs
#: a ``str``, so the ``TypeError`` was raised while the f-string was being built -- inside the
#: caller, before any reporting code ran -- and the real error was discarded. Nothing reported
#: through ``on_event`` or the console may contain this text: that is the regression guard.
ESCAPE_TYPE_ERROR = "expected string or bytes-like object"


def drive(
    client: StubLLMClient,
    goal: str = "GOAL",
    on_event: Callable[[str], None] | None = None,
) -> tuple[Orchestrator, list[str]]:
    """Run a full orchestrator pass over ``client`` and return it with its events."""
    events: list[str] = []

    async def scenario() -> Orchestrator:
        orchestrator = Orchestrator(goal=goal, client=client, on_event=on_event or events.append)
        await orchestrator.run()
        return orchestrator

    return asyncio.run(scenario()), events


def limit_cycles(monkeypatch: pytest.MonkeyPatch, cycles: int) -> None:
    monkeypatch.setattr(singleton_settings, "max_cycles", cycles)


def limit_parallelism(monkeypatch: pytest.MonkeyPatch, limit: int) -> None:
    monkeypatch.setattr(singleton_settings, "max_parallel_tasks", limit)


def statuses(orchestrator: Orchestrator) -> list[tuple[str, str]]:
    return [(task.id, task.status) for task in orchestrator.tasks]


# --------------------------------------------------------- the loop does not stop early


def test_the_run_keeps_cycling_until_the_goal_is_really_met(unlimited_cycles: None) -> None:
    # Regression: _get_tasks() used to return [], which made _is_done() trivially true, so the
    # loop halted after a single cycle even though the suite was still red.
    client = StubLLMClient(verdicts=[FAILING_VERDICT, FAILING_VERDICT, PASSING_VERDICT])

    orchestrator, events = drive(client)

    assert orchestrator.cycles == 3
    assert orchestrator.last_test_passed is True
    assert events[-1] == "Goal complete: every task is done and tests pass. Halting."


def test_completion_requires_a_passing_test_suite(monkeypatch: pytest.MonkeyPatch) -> None:
    # Every planned task completes, but the tester keeps saying the suite is red: the loop
    # must keep going and must never report success.
    limit_cycles(monkeypatch, 2)
    client = StubLLMClient(verdicts=[FAILING_VERDICT, FAILING_VERDICT])

    orchestrator, events = drive(client)

    assert orchestrator.cycles == 2
    assert orchestrator.last_test_passed is False
    assert all(task.status == "completed" for task in orchestrator.tasks)
    assert not any("Goal complete" in event for event in events)
    assert "Cycle limit reached (max_cycles=2). Halting." in events[-1]


def test_max_cycles_zero_means_unlimited(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 0)
    client = StubLLMClient(
        verdicts=[FAILING_VERDICT, FAILING_VERDICT, FAILING_VERDICT, PASSING_VERDICT]
    )

    orchestrator, _ = drive(client)

    assert orchestrator.cycles == 4


def test_max_cycles_is_honoured_for_an_unachievable_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 2)
    client = StubLLMClient(verdicts=[FAILING_VERDICT] * 20)

    orchestrator, events = drive(client)

    assert orchestrator.cycles == 2
    assert len(client.planner_prompts) == 2
    assert events[-1] == "Cycle limit reached (max_cycles=2). Halting."


# ------------------------------------------------------------------------ empty plans


def test_an_empty_task_list_is_not_done() -> None:
    client = StubLLMClient()

    assert Orchestrator(goal="GOAL", client=client)._is_done() is False


def test_all_tasks_completed_but_tests_red_is_not_done(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    orchestrator, _ = drive(StubLLMClient(verdicts=[FAILING_VERDICT]))

    assert all(task.status == "completed" for task in orchestrator.tasks)
    assert orchestrator.last_test_passed is False
    assert orchestrator._is_done() is False

    orchestrator.last_test_passed = True
    assert orchestrator._is_done() is True


def test_a_plan_that_cannot_be_parsed_halts_instead_of_looping(
    console_output: io.StringIO,
) -> None:
    # An unusable response is a hard planning failure: the run must terminate after one cycle
    # and never claim success. It must terminate *by reporting* the planner's own error, not
    # by propagating one -- see ESCAPE_TYPE_ERROR for the crash this used to pin instead.
    client = StubLLMClient(plans=["I am afraid I cannot help with that."])
    events: list[str] = []

    orchestrator = Orchestrator(goal="GOAL", client=client, on_event=events.append)

    # No pytest.raises: run() returns normally, having halted on the planning failure.
    asyncio.run(orchestrator.run())

    assert orchestrator.cycles == 1
    assert orchestrator.last_test_passed is False
    assert orchestrator.tasks == []
    # The genuine planner error reaches the event stream, attributed to the planner and quoting
    # the offending response, rather than a TypeError from the reporting code.
    reported = [event for event in events if event.startswith("Planner failed: ")]
    assert len(reported) == 1
    assert "Planner returned no usable plan" in reported[0]
    assert "I am afraid I cannot help with that." in reported[0]
    # Terminal reason for a UI that has to explain why nothing is happening any more.
    assert events[-1] == "Halting: planning failed, cannot continue."
    assert not any("Goal complete" in event for event in events)
    assert ESCAPE_TYPE_ERROR not in "\n".join(events)
    assert ESCAPE_TYPE_ERROR not in console_output.getvalue()


def test_a_plan_that_makes_no_progress_stops_at_the_stall_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A Plan that carries no tasks at all (which PlannerAgent will not emit, but the Plan
    # model permits) exercises the no-progress guard: the run must terminate on its own
    # rather than cycle forever.
    limit_cycles(monkeypatch, 0)
    client = StubLLMClient(verdicts=[FAILING_VERDICT] * 20)

    async def empty_plan(goal: str) -> Plan:
        return Plan(tasks=[], rationale="nothing to do")

    events: list[str] = []

    async def scenario() -> Orchestrator:
        orchestrator = Orchestrator(goal="GOAL", client=client, on_event=events.append)
        orchestrator.planner.plan = empty_plan  # type: ignore[method-assign]
        await orchestrator.run()
        return orchestrator

    orchestrator = asyncio.run(scenario())

    stall = Orchestrator.MAX_STALLED_CYCLES
    # The guard counts consecutive cycles that did not move the *outcome* signature, and that
    # signature is keyed on the completed/failed task descriptions plus the current test
    # failures. Cycle 1 genuinely moves it -- a red suite has no failure signature until the
    # tester has actually run -- so the streak only starts at cycle 2 and the run needs
    # MAX_STALLED_CYCLES unchanged cycles *after* that first change. The total is therefore a
    # property of this scenario rather than of the constant, so assert the invariant: the run
    # halted, it halted *because* of the stall guard, and it did so in a bounded number of
    # cycles derived from MAX_STALLED_CYCLES.
    assert stall + 1 <= orchestrator.cycles <= 2 * stall
    assert orchestrator.tasks == []
    stall_event = f"Halting: no progress in {stall} consecutive cycles."
    assert stall_event in events
    assert events[-1] == stall_event
    assert not any("Goal complete" in event for event in events)
    # Bounded work, not a runaway: the planner is stubbed out above, so the only calls that
    # reach the client are one tester call per cycle and there is nothing to execute. Keying
    # the signature on the task list instead made the guard unreachable, and a run that keeps
    # re-planning the same work burned 335 cycles / 1002 LLM calls under max_cycles=0.
    assert len(client.calls) == orchestrator.cycles
    assert len(client.tester_prompts) == orchestrator.cycles
    assert client.worker_prompts == []


# --------------------------------------------------------------------- parallelism cap


@pytest.mark.parametrize(("limit", "task_count"), [(1, 5), (2, 6), (3, 7)])
def test_max_parallel_tasks_bounds_real_concurrency(
    monkeypatch: pytest.MonkeyPatch, limit: int, task_count: int
) -> None:
    limit_cycles(monkeypatch, 1)
    limit_parallelism(monkeypatch, limit)
    batch = plan_json(*[task_json(f"t{i}", f"Task {i}") for i in range(task_count)])
    client = StubLLMClient(plans=[batch], verdicts=[FAILING_VERDICT])

    orchestrator, _ = drive(client)

    # peak_concurrency is measured inside the stub with a live counter, so this is the real
    # simultaneous fan-out and not an artefact of how gather happens to schedule.
    assert client.peak_concurrency <= limit
    assert client.peak_concurrency == limit, "the batch should saturate the semaphore"
    assert len(orchestrator.tasks) == task_count
    assert all(task.status == "completed" for task in orchestrator.tasks)


def test_all_tasks_in_a_batch_are_actually_run(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    limit_parallelism(monkeypatch, 2)
    batch = plan_json(*[task_json(f"t{i}", f"Task {i}") for i in range(6)])
    client = StubLLMClient(plans=[batch], verdicts=[FAILING_VERDICT])

    drive(client)

    assert len(client.worker_prompts) == 6
    assert client.worker_order == [f"Task {i}" for i in range(6)]


def test_a_non_positive_parallel_limit_is_treated_as_serial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit_cycles(monkeypatch, 1)
    limit_parallelism(monkeypatch, 0)
    batch = plan_json(*[task_json(f"t{i}", f"Task {i}") for i in range(4)])
    client = StubLLMClient(plans=[batch], verdicts=[FAILING_VERDICT])

    drive(client)

    assert client.peak_concurrency == 1
    assert len(client.worker_prompts) == 4


# ---------------------------------------------------------------------- dependencies


def test_a_dependent_task_runs_only_after_its_dependency_completes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit_cycles(monkeypatch, 1)
    # "2" is listed first on purpose: the plan order must not be the execution order.
    batch = plan_json(
        task_json("2", "SECOND", dependencies=["1"]),
        task_json("1", "FIRST"),
    )
    client = StubLLMClient(plans=[batch], verdicts=[PASSING_VERDICT])

    orchestrator, _ = drive(client)

    assert client.worker_order == ["FIRST", "SECOND"]
    assert statuses(orchestrator) == [("2", "completed"), ("1", "completed")]


def test_a_chain_of_three_dependencies_runs_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    batch = plan_json(
        task_json("c", "THIRD", dependencies=["b"]),
        task_json("b", "SECOND", dependencies=["a"]),
        task_json("a", "FIRST"),
    )
    client = StubLLMClient(plans=[batch], verdicts=[PASSING_VERDICT])

    drive(client)

    assert client.worker_order == ["FIRST", "SECOND", "THIRD"]


def test_independent_tasks_still_run_together(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    limit_parallelism(monkeypatch, 3)
    batch = plan_json(
        task_json("a", "A"),
        task_json("b", "B"),
        task_json("c", "C", dependencies=["a"]),
    )
    client = StubLLMClient(plans=[batch], verdicts=[PASSING_VERDICT])

    drive(client)

    # "a" and "b" are independent so they overlap; "c" waits for "a".
    assert client.peak_concurrency == 2
    assert client.worker_order == ["A", "B", "C"]


def test_ready_batch_excludes_tasks_with_unfinished_dependencies() -> None:
    client = StubLLMClient()
    orchestrator = Orchestrator(goal="GOAL", client=client)
    first = TaskModel(id="1", description="A")
    second = TaskModel(id="2", description="B", dependencies=["1"])
    orchestrator.tasks = [first, second]

    assert orchestrator._ready_batch() == [first]

    first.status = "completed"
    assert orchestrator._ready_batch() == [second]


def test_ready_batch_ignores_dependencies_that_do_not_exist() -> None:
    client = StubLLMClient()
    orchestrator = Orchestrator(goal="GOAL", client=client)
    only = TaskModel(id="1", description="A", dependencies=["ghost"])
    orchestrator.tasks = [only]

    assert orchestrator._ready_batch() == [only]


def test_ready_batch_skips_tasks_that_are_not_pending() -> None:
    client = StubLLMClient()
    orchestrator = Orchestrator(goal="GOAL", client=client)
    orchestrator.tasks = [
        TaskModel(id="1", description="A", status="completed"),
        TaskModel(id="2", description="B", status="failed"),
    ]

    assert orchestrator._ready_batch() == []


def test_a_pending_task_with_a_failed_dependency_is_reported_as_skipped() -> None:
    client = StubLLMClient()
    events: list[str] = []
    orchestrator = Orchestrator(goal="GOAL", client=client, on_event=events.append)
    orchestrator.tasks = [
        TaskModel(id="1", description="A", status="failed", result="boom"),
        TaskModel(id="2", description="B", dependencies=["1"]),
        TaskModel(id="3", description="C"),
    ]

    orchestrator._report_blocked()

    assert events == ["Task 2 skipped: dependency 1 did not complete."]


def test_a_task_whose_dependency_failed_is_never_run(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    batch = plan_json(
        task_json("1", "BOOM"),
        task_json("2", "NEVER RUNS", dependencies=["1"]),
    )

    def explode(prompt: str) -> BaseException:
        if "BOOM" in prompt:
            return RuntimeError("kaboom")
        return None

    client = StubLLMClient(plans=[batch], verdicts=[FAILING_VERDICT], worker_hook=explode)
    events: list[str] = []

    orchestrator = Orchestrator(goal="GOAL", client=client, on_event=events.append)
    asyncio.run(orchestrator.run())

    assert statuses(orchestrator) == [("1", "failed"), ("2", "pending")]
    assert "NEVER RUNS" not in client.worker_prompts
    assert "Task 2 skipped: dependency 1 did not complete." in events


# --------------------------------------------------------------- goal and re-planning


def test_a_test_failure_never_rewrites_the_original_goal(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 2)
    client = StubLLMClient(verdicts=[FAILING_VERDICT, FAILING_VERDICT, PASSING_VERDICT])

    orchestrator, _ = drive(client, goal="Add JWT auth")

    assert orchestrator.original_goal == "Add JWT auth"
    assert orchestrator.goal == "Add JWT auth"


def test_the_next_planning_prompt_carries_the_goal_and_the_failure_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit_cycles(monkeypatch, 2)
    client = StubLLMClient(verdicts=[FAILING_VERDICT, PASSING_VERDICT])

    drive(client, goal="Add JWT auth")

    assert len(client.planner_prompts) == 2
    second = client.planner_prompts[1]
    assert "Add JWT auth" in second
    assert "tests/test_widget.py::test_build" in second
    assert "The previous cycle made no clean finish." in second


def test_the_planning_goal_is_the_bare_goal_when_nothing_went_wrong() -> None:
    client = StubLLMClient()
    orchestrator = Orchestrator(goal="Add JWT auth", client=client)

    assert orchestrator._planning_goal() == "Add JWT auth"


def test_the_planning_goal_lists_failed_tasks_and_test_failures() -> None:
    client = StubLLMClient()
    orchestrator = Orchestrator(goal="Add JWT auth", client=client)
    orchestrator.tasks = [TaskModel(id="1", description="A", status="failed")]
    orchestrator.test_failures = ["first failure", "second failure"]

    goal = orchestrator._planning_goal()

    assert goal.startswith("Add JWT auth")
    assert "- Tasks that failed: 1" in goal
    assert "- Test summary: first failure" in goal
    assert "- Test failure: second failure" in goal


def test_a_completed_task_is_never_re_run(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    client = StubLLMClient(plans=[ONE_TASK], verdicts=[FAILING_VERDICT])

    async def scenario() -> Orchestrator:
        orchestrator = Orchestrator(goal="GOAL", client=client)
        done = TaskModel(id="a", description="Do A", status="completed", result="kept")
        orchestrator.tasks.append(done)
        await orchestrator.run()
        return orchestrator

    orchestrator = asyncio.run(scenario())

    assert orchestrator.tasks[0].status == "completed"
    assert orchestrator.tasks[0].result == "kept"
    assert [task.id for task in orchestrator.tasks] == ["a", "a-2"]
    # Exactly one worker call: the pre-completed task was left alone, only the new "a-2" ran.
    assert len(client.worker_prompts) == 1


def test_plans_accumulate_across_cycles_and_duplicate_ids_are_suffixed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit_cycles(monkeypatch, 2)
    client = StubLLMClient(plans=[ONE_TASK], verdicts=[FAILING_VERDICT])

    orchestrator, events = drive(client)

    assert [task.id for task in orchestrator.tasks] == ["a", "a-2"]
    assert all(task.status == "completed" for task in orchestrator.tasks)
    assert "Plan reused id 'a'; new task registered as 'a-2'." in events
    assert len(client.planner_prompts) == 2


def test_a_worker_result_is_stored_on_its_task(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    client = StubLLMClient(
        plans=[ONE_TASK], verdicts=[FAILING_VERDICT], summaries=["touched 2 files"]
    )

    orchestrator, _ = drive(client)

    assert orchestrator.tasks[0].result == "touched 2 files"


# ------------------------------------------------------------------------- events


def test_events_are_non_empty_plain_text(unlimited_cycles: None) -> None:
    client = StubLLMClient(verdicts=[PASSING_VERDICT])

    _, events = drive(client, goal="Add JWT auth")

    assert events
    for event in events:
        assert isinstance(event, str)
        assert event.strip(), repr(event)
        for tag in RICH_TAGS:
            assert tag not in event, f"{tag!r} leaked into {event!r}"


def test_events_describe_every_stage_of_a_cycle(unlimited_cycles: None) -> None:
    client = StubLLMClient(verdicts=[PASSING_VERDICT])

    _, events = drive(client, goal="Add JWT auth")

    joined = "\n".join(events)
    assert "Goal: Add JWT auth" in joined
    assert "Cycle 1" in joined
    assert "Plan: 1 new task(s)" in joined
    assert "Task a completed" in joined
    assert "Tests passed: all tests passed" in joined


def test_the_completion_reason_is_always_emitted(unlimited_cycles: None) -> None:
    client = StubLLMClient(verdicts=[PASSING_VERDICT])

    orchestrator, events = drive(client)

    assert events[-1] == "Goal complete: every task is done and tests pass. Halting."
    assert orchestrator.last_test_passed is True


def test_the_cycle_limit_reason_is_always_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    client = StubLLMClient(verdicts=[FAILING_VERDICT] * 5)

    _, events = drive(client)

    assert events[-1] == "Cycle limit reached (max_cycles=1). Halting."


def test_an_empty_model_task_list_is_a_planning_failure(console_output: io.StringIO) -> None:
    # PlannerAgent refuses to build a Plan out of zero usable tasks, and the orchestrator treats
    # that as fatal: one cycle, the planner's own message, a clean halt. The failure is reported,
    # not raised -- an (TypeError, ValueError) escaping run() was the old behaviour.
    client = StubLLMClient(plans=['{"tasks": [], "rationale": "nothing to do"}'])
    events: list[str] = []
    orchestrator = Orchestrator(goal="GOAL", client=client, on_event=events.append)

    asyncio.run(orchestrator.run())

    assert orchestrator.cycles == 1
    assert orchestrator.tasks == []
    reported = [event for event in events if event.startswith("Planner failed: ")]
    assert len(reported) == 1
    assert "the plan contains no usable tasks" in reported[0]
    assert '{"tasks": [], "rationale": "nothing to do"}' in reported[0]
    assert events[-1] == "Halting: planning failed, cannot continue."
    assert not any("Goal complete" in event for event in events)
    assert ESCAPE_TYPE_ERROR not in "\n".join(events)
    assert ESCAPE_TYPE_ERROR not in console_output.getvalue()


def test_the_stall_reason_is_always_emitted(unlimited_cycles: None) -> None:
    client = StubLLMClient(verdicts=[FAILING_VERDICT] * 20)

    async def scenario() -> list[str]:
        events: list[str] = []

        async def empty_plan(goal: str) -> Plan:
            return Plan(tasks=[], rationale="nothing")

        orchestrator = Orchestrator(goal="GOAL", client=client, on_event=events.append)
        orchestrator.planner.plan = empty_plan  # type: ignore[method-assign]
        await orchestrator.run()
        return events

    events = asyncio.run(scenario())

    assert events[-1] == (
        f"Halting: no progress in {Orchestrator.MAX_STALLED_CYCLES} consecutive cycles."
    )


def test_an_event_callback_that_raises_does_not_break_the_run(unlimited_cycles: None) -> None:
    def explode(event: str) -> None:
        raise RuntimeError("UI bug")

    client = StubLLMClient(verdicts=[PASSING_VERDICT])

    orchestrator, _ = drive(client, on_event=explode)

    assert orchestrator.last_test_passed is True


def test_a_failing_verdict_reports_every_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    client = StubLLMClient(verdicts=[FAILING_VERDICT])

    orchestrator, events = drive(client)

    assert orchestrator.last_test_passed is False
    assert orchestrator.test_failures == [
        "tests/test_widget.py::test_build AssertionError: expected 3"
    ]
    assert "Tests failed: 1 failed, 1 passed" in events
    assert "Will attempt a fix in the next cycle." in events


# ----------------------------------------------------------------------- markup safety


def test_a_goal_full_of_markup_does_not_crash_the_run(unlimited_cycles: None) -> None:
    client = StubLLMClient(verdicts=[PASSING_VERDICT])

    orchestrator, _ = drive(client, goal="Fix the [/] markup")

    assert orchestrator.cycles == 1
    assert orchestrator.last_test_passed is True


@pytest.mark.parametrize(
    "goal",
    [
        "Fix the [/] markup",
        "Support [bold]text[/bold] in the [red]renderer[/red]",
        "Escape [link=http://x]this[/link] please",
        "A [not-a-tag] in the middle",
        "Brackets [] and [[nested]]",
    ],
)
def test_the_goal_event_preserves_the_users_characters(unlimited_cycles: None, goal: str) -> None:
    client = StubLLMClient(verdicts=[PASSING_VERDICT])

    orchestrator, events = drive(client, goal=goal)

    assert events[0] == f"Goal: {goal}"
    assert orchestrator.original_goal == goal


def test_untrusted_test_output_cannot_inject_markup(
    monkeypatch: pytest.MonkeyPatch, console_output: io.StringIO
) -> None:
    verdict = json.dumps(
        {
            "passed": False,
            "summary": "closing tag [/] sneaks in",
            "failures": ["[bold red]not really bold[/bold red]", "[/]"],
        }
    )
    limit_cycles(monkeypatch, 1)
    client = StubLLMClient(verdicts=[verdict])

    _, events = drive(client, goal="GOAL")

    # The summary reaches on_event with the user's characters intact ...
    assert "Tests failed: closing tag [/] sneaks in" in events
    # ... and the individual failure lines reach the console without being reinterpreted.
    printed = console_output.getvalue()
    assert "[bold red]not really bold[/bold red]" in printed


def test_untrusted_task_ids_cannot_inject_markup(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    batch = plan_json(task_json("[/]evil[/]", "Do the thing"))
    client = StubLLMClient(plans=[batch], verdicts=[PASSING_VERDICT])

    orchestrator, events = drive(client)

    assert orchestrator.tasks[0].id == "[/]evil[/]"
    assert "Task [/]evil[/] completed" in events


# ------------------------------------------------------------------- failure handling


def test_a_planner_exception_halts_gracefully(unlimited_cycles: None) -> None:
    client = StubLLMClient(planner_error=RuntimeError("planner exploded"))

    orchestrator, events = drive(client)

    assert orchestrator.cycles == 1
    assert any("Planner failed" in event for event in events)
    assert events[-1] == "Halting: planning failed, cannot continue."
    assert orchestrator.last_test_passed is False


def test_a_tester_exception_is_reported_as_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    limit_cycles(monkeypatch, 1)
    client = StubLLMClient(tester_error=RuntimeError("tester exploded"))

    orchestrator, events = drive(client)

    assert orchestrator.last_test_passed is False
    assert orchestrator.test_failures == ["Test run raised RuntimeError: tester exploded"]
    assert any("Test run raised an error" in event for event in events)


def test_a_successful_worker_result_is_recorded_on_the_task() -> None:
    client = StubLLMClient()
    orchestrator = Orchestrator(goal="GOAL", client=client)
    task = TaskModel(id="1", description="A")

    orchestrator._record(task, "a summary")

    assert task.status == "completed"
    assert task.result == "a summary"


def test_cancellation_propagates_out_of_run_and_leaves_the_task_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limit_cycles(monkeypatch, 1)
    batch = plan_json(task_json("a", "HANG"))

    def cancel(prompt: str) -> BaseException | None:
        return asyncio.CancelledError() if "HANG" in prompt else None

    client = StubLLMClient(plans=[batch], worker_hook=cancel)

    async def scenario(orchestrator: Orchestrator) -> None:
        await orchestrator.run()

    orchestrator = Orchestrator(goal="GOAL", client=client)
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario(orchestrator))

    # The cancelled task was planned but never recorded, so it is still pending, not done.
    assert statuses(orchestrator) == [("a", "pending")]
    assert orchestrator.tasks[0].result is None
    assert orchestrator.last_test_passed is False


def test_a_cancelled_task_in_a_batch_is_re_raised_and_records_nothing() -> None:
    batch = plan_json(task_json("a", "FINE"), task_json("b", "HANG"))

    def cancel(prompt: str) -> BaseException | None:
        return asyncio.CancelledError() if "HANG" in prompt else None

    client = StubLLMClient(plans=[batch], worker_hook=cancel)
    orchestrator = Orchestrator(goal="GOAL", client=client)
    orchestrator.tasks = [
        TaskModel(id="a", description="FINE"),
        TaskModel(id="b", description="HANG"),
    ]

    async def scenario() -> None:
        await orchestrator._execute()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(scenario())

    # gather(return_exceptions=True) hands the CancelledError back and _execute re-raises it
    # while walking the results, so the task that finished is recorded and the cancelled one
    # never is: a cancelled task must never come back as "completed".
    assert statuses(orchestrator) == [("a", "completed"), ("b", "pending")]
    assert orchestrator.tasks[1].result is None


# ------------------------------------------------------------------------- defaults


def test_an_orchestrator_builds_its_agents_from_one_client() -> None:
    client = StubLLMClient()
    orchestrator = Orchestrator(goal="GOAL", client=client)

    assert orchestrator.planner.client is client
    assert orchestrator.original_goal == "GOAL"
    assert orchestrator.tasks == []
    assert orchestrator.cycles == 0
    assert orchestrator.last_test_passed is False
    assert orchestrator.test_failures == []


def test_each_role_uses_its_own_configured_model(unlimited_cycles: None) -> None:
    settings = Settings(planner_model="p-model", worker_model="w-model", tester_model="t-model")
    client = StubLLMClient(settings=settings, verdicts=[PASSING_VERDICT])

    drive(client)

    models = {call.model for call in client.calls}
    assert models == {"p-model", "w-model", "t-model"}


def test_progress_resets_the_stall_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    # Two idle cycles then real progress: without the reset the guard would have halted at
    # cycle MAX_STALLED_CYCLES, so reaching the cycle limit proves it was reset.
    limit_cycles(monkeypatch, 5)
    client = StubLLMClient(verdicts=[FAILING_VERDICT] * 5)
    sequence = [
        Plan(tasks=[], rationale="nothing"),
        Plan(tasks=[], rationale="nothing"),
        Plan(tasks=[TaskModel(id="a", description="Do A")], rationale="r"),
    ]
    remaining = iter(sequence)

    async def next_plan(goal: str) -> Plan:
        return next(remaining, sequence[-1])

    async def scenario() -> tuple[Orchestrator, list[str]]:
        events: list[str] = []
        orchestrator = Orchestrator(goal="GOAL", client=client, on_event=events.append)
        orchestrator.planner.plan = next_plan  # type: ignore[method-assign]
        await orchestrator.run()
        return orchestrator, events

    orchestrator, events = asyncio.run(scenario())

    assert orchestrator.cycles == 5
    assert events[-1] == "Cycle limit reached (max_cycles=5). Halting."
    assert "no progress" not in events[-1]
