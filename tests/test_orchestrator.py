"""The Furrow loop: scheduling, termination and the event stream.

Only two collaborators are faked. ``PlannerAgent`` is replaced with a scripted stand-in
so a test can say exactly what each cycle should plan, and the LLM client is faked so
the real ``WorkerAgent`` and ``TesterAgent`` run with scripted answers. The
orchestrator, its scheduler, its termination logic and its event stream are the real
implementation throughout.

The headline test is
:func:`test_more_than_one_cycle_runs_when_the_planner_keeps_finding_work`: the task
lookup used to return a hardcoded empty list, so ``0 >= 0`` ended every run after
exactly one cycle with "Goal complete. Halting."
"""

from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from rich.console import Console

from furrow.agents.planner import PlannerAgent
from furrow.config import Plan, TaskModel
from furrow.config import TestResult as Verdict
from furrow.core.orchestrator import (
    MAX_CONSECUTIVE_CYCLE_ERRORS,
    MAX_EMPTY_PLAN_STREAK,
    MAX_TASK_ATTEMPTS,
    Orchestrator,
)
from furrow.llm import LLMClient

from .conftest import PASSING_SUITE, FakeLLM, SettingsFactory, run_coroutine

GOAL = "add a widget to the dashboard"

#: Every event type the orchestrator is documented to emit, and nothing else.
DOCUMENTED_EVENT_TYPES = frozenset(
    {
        "run_start",
        "cycle_start",
        "plan",
        "task_start",
        "task_done",
        "task_failed",
        "tests",
        "cycle_done",
        "warning",
        "error",
        "run_done",
    }
)

PASSED = Verdict(passed=True, summary="1 passed")
FAILED = Verdict(
    passed=False,
    summary="1 failed",
    failures=["tests/test_widget.py::test_render - AssertionError: expected 2, got 1"],
)

EMPTY_PLAN = Plan(tasks=[], rationale="nothing left to do")

RunFactory = Callable[..., "Run"]


def task(task_id: str, depends_on: Sequence[str] = ()) -> TaskModel:
    """A task whose description doubles as the worker's prompt marker."""
    return TaskModel(id=task_id, description=f"do {task_id}", dependencies=list(depends_on))


def plan(*tasks: TaskModel, rationale: str = "one slice at a time") -> Plan:
    return Plan(tasks=list(tasks), rationale=rationale)


class ScriptedPlanner:
    """Stand-in for :class:`PlannerAgent` that replays one outcome per cycle."""

    def __init__(self, outcomes: Sequence[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[tuple[str, str]] = []

    async def plan(self, goal: str, feedback: str = "", context: str = "") -> Plan:
        self.calls.append((goal, feedback))
        outcome = self.outcomes.pop(0) if self.outcomes else EMPTY_PLAN
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


@dataclass
class WorkerLog:
    """What every worker invocation did, and how many ran at once."""

    outcomes: Mapping[str, Any]
    timeline: list[tuple[str, str]] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    in_flight: int = 0
    peak: int = 0

    @property
    def started(self) -> list[str]:
        return [task_id for action, task_id in self.timeline if action == "start"]

    @property
    def finished(self) -> list[str]:
        return [task_id for action, task_id in self.timeline if action == "finish"]

    async def __call__(self, prompt: str, client: FakeLLM) -> str:
        task_id = _worker_task_id(prompt)
        self.prompts.append(prompt)
        self.timeline.append(("start", task_id))
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            outcome = self.outcomes.get(task_id, f"described the change for {task_id}")
            if callable(outcome):
                outcome = outcome()
            if isinstance(outcome, BaseException):
                raise outcome
            return cast(str, outcome)
        finally:
            self.in_flight -= 1
            self.timeline.append(("finish", task_id))


@dataclass
class Run:
    """An orchestrator plus everything a test needs to inspect after ``run()``."""

    orchestrator: Orchestrator
    planner: ScriptedPlanner
    client: FakeLLM
    workers: WorkerLog
    events: list[dict[str, Any]] = field(default_factory=list)
    tester_prompts: list[str] = field(default_factory=list)

    @property
    def cycles(self) -> int:
        return self.orchestrator.cycles

    @property
    def done(self) -> bool:
        return self.orchestrator.done

    def of(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["type"] == kind]

    def run_done(self) -> dict[str, Any]:
        finals = self.of("run_done")
        assert finals, f"no run_done event; saw {[event['type'] for event in self.events]}"
        return finals[-1]

    def messages(self, kind: str) -> str:
        return "\n".join(event["message"] for event in self.of(kind))

    def planned(self, task_id: str) -> TaskModel:
        for candidate in self.orchestrator.tasks:
            if candidate.id == task_id:
                return candidate
        raise AssertionError(f"task {task_id!r} was never planned")

    def result_of(self, task_id: str) -> str:
        return self.planned(task_id).result or ""


def _worker_task_id(prompt: str) -> str:
    for line in prompt.splitlines():
        if line.startswith("Task: do "):
            return line.removeprefix("Task: do ")
    raise AssertionError("the worker prompt carried no task description")


@pytest.fixture
def make_run(
    make_settings: SettingsFactory,
    make_client: Callable[..., FakeLLM],
    suite_command: Callable[..., str],
) -> RunFactory:
    """Build an orchestrator with a scripted planner, workers and tester verdicts."""

    def _make(
        plans: Sequence[Any],
        *,
        workers: Mapping[str, Any] | None = None,
        verdicts: Sequence[Verdict] | None = None,
        overrides: Mapping[str, Any] | None = None,
        goal: str = GOAL,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> Run:
        values: dict[str, Any] = {"test_command": suite_command(PASSING_SUITE)}
        values.update(overrides or {})
        settings = make_settings(**values)
        worker_log = WorkerLog(dict(workers or {}))
        verdict_script = list(verdicts or [PASSED])
        tester_prompts: list[str] = []

        def dispatch(prompt: str, client: FakeLLM) -> str | Awaitable[str]:
            if "planning agent" in prompt:
                raise AssertionError("the scripted planner must not reach the client")
            if "tester agent" in prompt:
                tester_prompts.append(prompt)
                index = len(tester_prompts) - 1
                verdict = verdict_script[min(index, len(verdict_script) - 1)]
                return json.dumps(
                    {
                        "passed": verdict.passed,
                        "summary": verdict.summary,
                        "failures": verdict.failures,
                    }
                )
            return worker_log(prompt, client)

        client = make_client(settings, handler=dispatch)
        events: list[dict[str, Any]] = []
        orchestrator = Orchestrator(
            goal=goal,
            client=cast(LLMClient, client),
            settings=settings,
            console=Console(file=io.StringIO()),
            on_event=events.append if on_event is None else on_event,
        )
        planner = ScriptedPlanner(plans)
        orchestrator.planner = cast(PlannerAgent, planner)
        return Run(
            orchestrator=orchestrator,
            planner=planner,
            client=client,
            workers=worker_log,
            events=events,
            tester_prompts=tester_prompts,
        )

    return _make


# ------------------------------------------------------------------- the loop


def test_more_than_one_cycle_runs_when_the_planner_keeps_finding_work(
    make_run: RunFactory,
) -> None:
    run = make_run([plan(task("a")), plan(task("b"))], verdicts=[FAILED, PASSED])
    run_coroutine(run.orchestrator.run())
    assert run.cycles >= 2
    assert run.workers.started == ["a", "b"]
    assert run.done is True
    assert run.run_done()["complete"] is True
    assert run.run_done()["cycles"] == 2


def test_a_completed_cycle_with_green_tests_ends_the_run_as_done(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"), task("b"))])
    run_coroutine(run.orchestrator.run())
    assert (run.cycles, run.done) == (1, True)
    assert run.run_done()["complete"] is True
    assert "completed" in run.run_done()["reason"]


def test_the_loop_keeps_going_while_the_tests_are_red(make_run: RunFactory) -> None:
    run = make_run(
        [plan(task("a")), plan(task("b")), plan(task("c"))],
        verdicts=[FAILED],
        overrides={"max_cycles": 3},
    )
    run_coroutine(run.orchestrator.run())
    assert run.cycles == 3
    assert run.done is False
    assert run.workers.started == ["a", "b", "c"]


def test_an_empty_plan_with_green_tests_completes_the_run(make_run: RunFactory) -> None:
    run = make_run([EMPTY_PLAN])
    run_coroutine(run.orchestrator.run())
    assert run.done is True
    assert "nothing left to do" in run.run_done()["reason"]


def test_the_goal_survives_a_failing_test_cycle(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"))], verdicts=[FAILED], overrides={"max_cycles": 1})
    run_coroutine(run.orchestrator.run())
    assert run.orchestrator.goal == GOAL


def test_every_planned_task_is_exposed_newest_last(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"), task("b"))])
    run_coroutine(run.orchestrator.run())
    assert [candidate.id for candidate in run.orchestrator.tasks] == ["a", "b"]


# --------------------------------------------------------------- termination


def test_cancelling_mid_run_never_reports_success(make_run: RunFactory) -> None:
    holder: list[Run] = []

    def on_event(event: dict[str, Any]) -> None:
        holder[0].events.append(event)
        if event["type"] == "cycle_done" and event.get("cycle") == 1:
            holder[0].orchestrator.cancel()

    run = make_run([plan(task("a")), plan(task("b"))], verdicts=[FAILED, PASSED], on_event=on_event)
    holder.append(run)
    run_coroutine(run.orchestrator.run())

    assert run.orchestrator.cancelled is True
    assert run.done is False
    assert run.cycles == 1
    assert run.workers.started == ["a"]
    assert run.run_done()["complete"] is False
    assert run.run_done()["cancelled"] is True
    assert "cancelled" in run.run_done()["message"].lower()


def test_a_run_cancelled_before_it_starts_runs_no_cycles(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"))])
    run.orchestrator.cancel()
    run_coroutine(run.orchestrator.run())
    assert (run.cycles, run.done) == (0, False)
    assert run.run_done()["cancelled"] is True


def test_cancelling_twice_announces_it_once(make_run: RunFactory) -> None:
    run = make_run([EMPTY_PLAN])
    run.orchestrator.cancel()
    run.orchestrator.cancel()
    run_coroutine(run.orchestrator.run())
    assert run.messages("warning").count("Cancellation requested") == 1


def test_max_cycles_stops_after_exactly_that_many_cycles(make_run: RunFactory) -> None:
    run = make_run(
        [plan(task(f"t{n}")) for n in range(5)],
        verdicts=[FAILED],
        overrides={"max_cycles": 2},
    )
    run_coroutine(run.orchestrator.run())
    assert run.cycles == 2
    assert run.done is False
    assert "max_cycles=2" in run.run_done()["reason"]


def test_the_circuit_breaker_stops_after_consecutive_errored_cycles(
    make_run: RunFactory,
) -> None:
    run = make_run([RuntimeError("planner is down")] * (MAX_CONSECUTIVE_CYCLE_ERRORS + 1))
    run_coroutine(run.orchestrator.run())
    assert run.cycles == MAX_CONSECUTIVE_CYCLE_ERRORS
    assert run.done is False
    assert "Circuit breaker" in run.run_done()["reason"]
    assert len(run.of("error")) == MAX_CONSECUTIVE_CYCLE_ERRORS


def test_a_clean_cycle_resets_the_circuit_breaker(make_run: RunFactory) -> None:
    run = make_run(
        [
            RuntimeError("one"),
            RuntimeError("two"),
            plan(task("a")),
            RuntimeError("three"),
            RuntimeError("four"),
            RuntimeError("five"),
        ],
        verdicts=[FAILED],
    )
    run_coroutine(run.orchestrator.run())
    assert run.cycles == 6
    assert "Circuit breaker" in run.run_done()["reason"]
    assert [event["cycle"] for event in run.of("error")] == [1, 2, 4, 5, 6]


def test_a_raising_cycle_is_reported_and_the_loop_continues(make_run: RunFactory) -> None:
    run = make_run([RuntimeError("planner is down"), plan(task("a"))])
    run_coroutine(run.orchestrator.run())
    errors = run.of("error")
    assert len(errors) == 1
    assert errors[0]["cycle"] == 1
    assert "RuntimeError: planner is down" in errors[0]["message"]
    assert run.cycles == 2
    assert run.done is True


def test_an_empty_plan_with_red_tests_stops_as_stagnation(make_run: RunFactory) -> None:
    run = make_run([EMPTY_PLAN] * (MAX_EMPTY_PLAN_STREAK + 2), verdicts=[FAILED])
    run_coroutine(run.orchestrator.run())
    assert run.cycles == MAX_EMPTY_PLAN_STREAK
    assert run.done is False
    assert "Stagnation" in run.run_done()["reason"]
    streaks = [event["streak"] for event in run.of("warning") if "streak" in event]
    assert streaks == [1, 2, 3]


def test_a_broken_event_sink_cannot_break_the_loop(make_run: RunFactory) -> None:
    def exploding_sink(event: dict[str, Any]) -> None:
        raise RuntimeError("the sink is broken")

    run = make_run([plan(task("a"))], on_event=exploding_sink)
    run_coroutine(run.orchestrator.run())
    assert run.cycles == 1
    assert run.done is True


# ---------------------------------------------------------------- scheduling


def test_max_parallel_tasks_caps_the_workers_in_flight(make_run: RunFactory) -> None:
    run = make_run(
        [plan(*(task(f"t{n}") for n in range(6)))],
        overrides={"max_parallel_tasks": 2},
    )
    run_coroutine(run.orchestrator.run())
    assert run.workers.peak == 2
    assert sorted(run.workers.finished) == [f"t{n}" for n in range(6)]


def test_independent_tasks_run_concurrently(make_run: RunFactory) -> None:
    run = make_run(
        [plan(*(task(f"t{n}") for n in range(4)))],
        overrides={"max_parallel_tasks": 4},
    )
    run_coroutine(run.orchestrator.run())
    assert run.workers.peak == 4


def test_a_dependent_task_starts_only_after_its_dependency_finishes(
    make_run: RunFactory,
) -> None:
    run = make_run([plan(task("a"), task("b", depends_on=["a"]))])
    run_coroutine(run.orchestrator.run())
    assert run.workers.timeline == [
        ("start", "a"),
        ("finish", "a"),
        ("start", "b"),
        ("finish", "b"),
    ]


def test_a_dependency_cycle_is_reported_without_launching_a_worker(
    make_run: RunFactory,
) -> None:
    run = make_run(
        [plan(task("a", depends_on=["b"]), task("b", depends_on=["a"]))],
        overrides={"max_cycles": 1},
    )
    run_coroutine(run.orchestrator.run())
    assert run.workers.started == []
    errors = run.of("error")
    assert len(errors) == 1
    assert "dependency cycle" in errors[0]["message"]
    assert len(run.of("task_failed")) == 2
    assert [candidate.status for candidate in run.orchestrator.tasks] == ["failed", "failed"]


def test_a_task_depending_on_itself_does_not_hang_the_run(make_run: RunFactory) -> None:
    run = make_run([plan(task("a", depends_on=["a"]))], overrides={"max_cycles": 1})
    run_coroutine(run.orchestrator.run())
    assert run.cycles == 1
    assert run.workers.started == []
    assert "dependency cycle" in run.messages("error")


# --------------------------------------------------------------------- retries


def test_a_failed_task_is_retried_then_abandoned_at_the_attempt_cap(
    make_run: RunFactory,
) -> None:
    run = make_run(
        [plan(task("a"))] * 4,
        workers={"a": lambda: RuntimeError("cannot do it")},
        verdicts=[FAILED],
        overrides={"max_cycles": 3},
    )
    run_coroutine(run.orchestrator.run())
    assert run.workers.started == ["a"] * MAX_TASK_ATTEMPTS
    assert run.planned("a").status == "failed"
    assert "RuntimeError: cannot do it" in run.result_of("a")
    assert f"all {MAX_TASK_ATTEMPTS} attempts have been used" in run.messages("warning")


def test_a_worker_cancelled_mid_flight_is_recorded_failed(make_run: RunFactory) -> None:
    run = make_run(
        [plan(task("a"))],
        workers={"a": lambda: asyncio.CancelledError()},
        overrides={"max_cycles": 1},
    )
    run_coroutine(run.orchestrator.run())
    assert run.planned("a").status == "failed"
    assert "CancelledError" in run.result_of("a")
    assert len(run.of("task_failed")) == 1
    assert run.done is False


def test_a_worker_exception_is_recorded_failed_with_its_message(
    make_run: RunFactory,
) -> None:
    run = make_run(
        [plan(task("a"))],
        workers={"a": lambda: RuntimeError("model refused")},
        overrides={"max_cycles": 1},
    )
    run_coroutine(run.orchestrator.run())
    assert run.planned("a").status == "failed"
    assert "model refused" in run.of("task_failed")[0]["message"]


def test_a_task_result_is_recorded_on_success(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"))], workers={"a": "a precise specification"})
    run_coroutine(run.orchestrator.run())
    assert run.planned("a").status == "completed"
    assert run.planned("a").result == "a precise specification"


# ------------------------------------------------------------------- feedback


def test_failing_test_output_reaches_the_next_cycle(make_run: RunFactory) -> None:
    run = make_run([plan(task("a")), plan(task("b"))], verdicts=[FAILED, PASSED])
    run_coroutine(run.orchestrator.run())
    assert "expected 2, got 1" in run.planner.calls[1][1]
    assert "expected 2, got 1" in run.workers.prompts[1]


def test_the_first_cycle_plans_without_feedback(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"))])
    run_coroutine(run.orchestrator.run())
    assert run.planner.calls[0] == (GOAL, "")


def test_the_tester_is_told_the_command_it_ran(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"))])
    run_coroutine(run.orchestrator.run())
    assert "Command: " in run.tester_prompts[0]
    assert "1 passed" in run.tester_prompts[0]


def test_a_failing_suite_is_reported_as_a_tests_event(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"))], verdicts=[FAILED], overrides={"max_cycles": 1})
    run_coroutine(run.orchestrator.run())
    verdicts = run.of("tests")
    assert verdicts[0]["passed"] is False
    assert verdicts[0]["failures"] == FAILED.failures


def test_the_last_test_result_is_only_set_once_the_tester_has_run(
    make_run: RunFactory,
) -> None:
    run = make_run([plan(task("a"))])
    assert run.orchestrator.last_test_result is None
    run_coroutine(run.orchestrator.run())
    assert run.orchestrator.last_test_result == PASSED


# ------------------------------------------------------------ client ownership


def test_an_injected_client_is_not_closed_by_the_run(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"))])
    run_coroutine(run.orchestrator.run())
    assert run.client.closed is False


def test_only_a_self_created_client_is_closed(
    make_settings: SettingsFactory,
    make_client: Callable[..., FakeLLM],
    suite_command: Callable[..., str],
) -> None:
    settings = make_settings(test_command=suite_command(PASSING_SUITE))
    client = make_client(settings)
    orchestrator = Orchestrator(goal=GOAL, settings=settings, console=Console(file=io.StringIO()))
    assert orchestrator._owns_client is True
    orchestrator.client = cast(LLMClient, client)
    orchestrator.planner = cast(PlannerAgent, ScriptedPlanner([EMPTY_PLAN]))
    run_coroutine(orchestrator.run())
    assert client.closed is True


def test_an_orchestrator_without_a_client_owns_the_one_it_made(
    make_settings: SettingsFactory, suite_command: Callable[..., str]
) -> None:
    settings = make_settings(test_command=suite_command(PASSING_SUITE))
    orchestrator = Orchestrator(goal=GOAL, settings=settings, console=Console(file=io.StringIO()))
    assert orchestrator._owns_client is True
    assert isinstance(orchestrator.client, LLMClient)
    assert orchestrator.client.settings is settings


# -------------------------------------------------------------- event contract


def test_the_event_stream_is_exactly_the_documented_set(make_run: RunFactory) -> None:
    working = make_run(
        [plan(task("a")), plan(task("b"))],
        workers={"b": lambda: asyncio.CancelledError()},
        verdicts=[FAILED, PASSED],
        overrides={"max_cycles": 2},
    )
    run_coroutine(working.orchestrator.run())

    deadlocked = make_run(
        [plan(task("a", depends_on=["b"]), task("b", depends_on=["a"]))],
        overrides={"max_cycles": 1},
    )
    run_coroutine(deadlocked.orchestrator.run())

    emitted = {event["type"] for event in working.events} | {
        event["type"] for event in deadlocked.events
    }
    assert emitted == DOCUMENTED_EVENT_TYPES


def test_every_event_carries_a_type_a_message_and_survives_json(
    make_run: RunFactory,
) -> None:
    run = make_run([plan(task("a"))])
    run_coroutine(run.orchestrator.run())
    assert all(event.get("type") and event.get("message") for event in run.events)
    assert all(json.dumps(event) for event in run.events)


def test_the_run_opens_with_the_goal(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"))])
    run_coroutine(run.orchestrator.run())
    assert run.of("run_start")[0]["goal"] == GOAL


def test_each_cycle_reports_its_number_and_its_tally(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"), task("b"))])
    run_coroutine(run.orchestrator.run())
    assert [event["cycle"] for event in run.of("cycle_start")] == [1]
    done = run.of("cycle_done")[0]
    assert (done["completed"], done["failed"], done["cycle"]) == (2, 0, 1)


def test_a_plan_event_lists_the_tasks_and_retries(make_run: RunFactory) -> None:
    run = make_run([plan(task("a"), task("b", depends_on=["a"]))])
    run_coroutine(run.orchestrator.run())
    event = run.of("plan")[0]
    assert [planned["id"] for planned in event["tasks"]] == ["a", "b"]
    assert event["retried"] == []


def test_a_retried_task_is_announced_in_the_next_plan_event(make_run: RunFactory) -> None:
    run = make_run(
        [plan(task("a")), plan(task("a"))],
        workers={"a": lambda: RuntimeError("nope")},
        verdicts=[FAILED],
        overrides={"max_cycles": 2},
    )
    run_coroutine(run.orchestrator.run())
    assert run.of("plan")[1]["retried"] == ["a"]
    assert [event["attempt"] for event in run.of("task_start")] == [1, 2]
