"""Offline tests for :mod:`furrow.core.orchestrator`.

Every test that drives :meth:`Orchestrator.run` or :meth:`Orchestrator._cycle`
takes the ``stub_tester`` fixture. Without it ``_cycle()`` would really shell out
to ``pytest -q``, ``ruff check .`` and ``mypy .`` inside the workspace, which is
both slow and completely unrelated to what is under test. The planner and the
workers are driven through :class:`tests.conftest.FakeLLM`, so no test can reach
a provider SDK either.

The headline test, :func:`test_run_keeps_cycling_instead_of_halting_after_one_cycle`,
is the regression this rewrite existed for: ``_get_tasks()`` used to return
``[]``, ``_is_done()`` therefore evaluated ``0 >= 0`` and ``run()`` announced
"Goal complete" after exactly one cycle regardless of what happened.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from rich.console import Console

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import (
    MAX_CONSECUTIVE_CYCLE_ERRORS,
    MAX_CONSECUTIVE_STALLED_CYCLES,
    Orchestrator,
)
from tests.conftest import FakeLLM, plan_reply, worker_reply

#: Placed at the end of a plan so the extract_json fence/prose path is exercised.
ONE_TASK = plan_reply(
    {"id": "1", "description": "write the thing", "files": ["thing.py"]},
)
OTHER_TASK = plan_reply(
    {"id": "2", "description": "write the other thing", "files": ["other.py"]},
)
NO_TASKS = plan_reply()

#: Registering an exception under the worker model is how a worker failure is
#: driven: ``WorkerAgent.run`` propagates it, ``asyncio.gather(return_exceptions
#: =True)`` turns it into ``status="failed"`` and the loop keeps going.
WORKER_BOOM = RuntimeError("worker exploded")
PLANNER_BOOM = RuntimeError("planner exploded")


class ScriptedReplies:
    """A canned reply that changes on every ``str()`` call.

    ``FakeLLM.complete`` returns ``str(value)`` for any canned value that is
    neither a ``str`` nor an exception instance, so registering one of these
    under a model yields a per-call sequence while still going through the real
    :class:`~furrow.agents.planner.PlannerAgent` and ``extract_json`` parsing.
    The last entry repeats once the script is exhausted.
    """

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.index = 0

    def __str__(self) -> str:
        reply = self._replies[min(self.index, len(self._replies) - 1)]
        self.index += 1
        return reply


class FixedPlanner:
    """A planner that returns a pre-built :class:`Plan` verbatim.

    Needed where the plan is larger than the concurrency cap under test: the real
    ``PlannerAgent`` clamps a plan to ``settings.max_parallel_tasks`` tasks, so a
    five-task plan cannot be produced while that cap is two. The scheduling
    guarantee being tested lives in ``Orchestrator._run_task``, not in the
    planner, so the plan is injected here instead of being shrunk.
    """

    def __init__(self, plan: Plan) -> None:
        # NOT ``self.plan``: an instance attribute would shadow the ``plan`` method
        # below and ``Orchestrator._cycle``'s ``await self.planner.plan(...)`` would
        # try to call the ``Plan`` object.
        self.fixed_plan = plan
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        return self.fixed_plan


class CancelAfterPlannerCalls:
    """Hook that raises ``CancelledError`` once ``after`` planner calls were seen.

    ``run()`` swallows ``Exception`` but not ``BaseException``, so this is how a
    deliberately unbounded (``max_cycles=0``) loop is stopped from a test with no
    timeout and no wall-clock dependency. It doubles as an assertion that a
    cancellation is never mistaken for a failed cycle.
    """

    def __init__(self, model: str, after: int) -> None:
        self.model = model
        self.after = after
        self.seen = 0

    async def __call__(self, *, model: str, prompt: str, system: str) -> None:
        if model != self.model:
            return
        self.seen += 1
        if self.seen >= self.after:
            raise asyncio.CancelledError


def passing_result(summary: str = "ok") -> TestResult:
    return TestResult(passed=True, summary=summary, failures=[])


def failing_result(*failures: str) -> TestResult:
    return TestResult(passed=False, summary="tests failed", failures=list(failures))


def planner_prompts(client: FakeLLM, model: str) -> list[str]:
    """Every prompt sent to ``model``, in call order."""
    return [prompt for call_model, prompt, _ in client.calls if call_model == model]


def task_id_in_prompt(prompt: str) -> str:
    """Read the ``Task id:`` line ``WorkerAgent.build_prompt`` always emits."""
    for section in prompt.split("\n\n"):
        if section.startswith("Task id: "):
            return section.removeprefix("Task id: ").split("\n", 1)[0]
    raise AssertionError("worker prompt carried no task id")


def console_output(console: Console) -> str:
    """Collapse all whitespace so soft-wrapped Rich output can be substring-matched."""
    return " ".join(str(console.file.getvalue()).split())


# --------------------------------------------------------------------- the loop


async def test_run_keeps_cycling_instead_of_halting_after_one_cycle(
    make_settings, silent_console, stub_tester
) -> None:
    # Regression: used to halt after one cycle with a bogus "Goal complete.",
    # because _get_tasks() always returned [] and _is_done() compared 0 >= 0.
    settings = make_settings(max_cycles=3)
    stub_tester([failing_result("tests failed")])
    client = FakeLLM(
        {
            # One task on the first two plans, nothing on the third: the run then
            # ends on the planner's empty-plan signal at cycle 3, not because the
            # loop gave up after a single pass and not by exhausting the budget.
            str(settings.planner_model): ScriptedReplies([ONE_TASK, OTHER_TASK, NO_TASKS]),
            # The worker never succeeds, so a task stays "failed" and the loop
            # has a reason to keep going.
            str(settings.worker_model): WORKER_BOOM,
        },
        settings=settings,
    )
    orchestrator = Orchestrator("keep going please", client, console=silent_console)

    await orchestrator.run()

    assert orchestrator.cycles >= 2, "the loop halted after a single cycle"
    assert orchestrator.cycles == 3, "the third plan ends the run only via max_cycles"

    # The assertion with teeth. _is_done() now short-circuits on an empty task
    # list, so monkeypatching _get_tasks to return [] no longer reproduces the
    # halt; what the old bug actually destroyed was *visibility* of the task
    # history, which is what made the loop unable to tell "finished" from
    # "stalled". These two assertions are the ones that fail when _get_tasks is
    # stubbed back to the buggy `return []`.
    visible = orchestrator._get_tasks()
    assert [task.id for task in visible] == ["1", "2"], "the loop lost its task history"
    assert [task.status for task in visible] == ["failed", "failed"]

    # The failure really was recorded rather than swallowed by the loop.
    assert all(str(WORKER_BOOM) in (task.result or "") for task in visible)
    assert "planner exploded" not in (visible[0].result or "")

    # It halted on the planner's "goal satisfied" signal, not on the cycle budget:
    # cycles == max_cycles here, so had the empty plan not halted the run, the
    # "Reached cycle limit" message would be in the output instead.
    assert orchestrator._no_work_remaining is True
    output = console_output(silent_console)
    assert "no further work" in output
    assert "Reached cycle limit" not in output

    # Regression guard for the false pass: the tasks failed *and* the suite is
    # red, so "Goal complete" must not be printed even though every cycle is over.
    assert "Goal complete" not in output, "a red run was reported as a success"


async def test_tasks_accumulate_across_cycles_and_get_tasks_returns_them(
    make_settings, silent_console, stub_tester
) -> None:
    # Task history: each cycle appends to the same list, and _get_tasks() hands
    # that very list back so _is_done() reasons over the whole run.
    settings = make_settings(max_cycles=2)
    stub_tester([failing_result("still broken")])
    client = FakeLLM(
        {
            str(settings.planner_model): ScriptedReplies([ONE_TASK, OTHER_TASK]),
            str(settings.worker_model): WORKER_BOOM,
        },
        settings=settings,
    )
    orchestrator = Orchestrator("accumulate", client, console=silent_console)

    await orchestrator.run()

    assert [task.id for task in orchestrator.tasks] == ["1", "2"]
    assert orchestrator._get_tasks() is orchestrator.tasks
    assert [task.status for task in orchestrator.tasks] == ["failed", "failed"]


# ------------------------------------------------------------------- _is_done()


def test_is_done_is_false_without_any_tasks(make_settings, silent_console) -> None:
    # A fresh orchestrator has planned nothing, so it is not "done" -- this is
    # the exact 0 >= 0 comparison that used to end the run after one cycle.
    orchestrator = Orchestrator("fresh", FakeLLM(settings=make_settings()), console=silent_console)

    assert orchestrator._get_tasks() == []
    assert orchestrator._is_done() is False


def test_is_done_is_true_only_when_every_task_completed(make_settings, silent_console) -> None:
    # At least one task and all of them completed is the single true case.
    orchestrator = Orchestrator("done", FakeLLM(settings=make_settings()), console=silent_console)
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]

    assert orchestrator._is_done() is True

    orchestrator.tasks = [TaskModel(id="1", description="a", status="completed")]
    assert orchestrator._is_done() is True


def test_is_done_is_false_when_any_task_failed(make_settings, silent_console) -> None:
    # One failure out of two is enough: the run must continue and retry.
    orchestrator = Orchestrator(
        "half done", FakeLLM(settings=make_settings()), console=silent_console
    )
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]

    assert orchestrator._is_done() is False


def test_is_done_is_false_while_a_task_is_still_pending(make_settings, silent_console) -> None:
    # A plan that has not been executed yet is pending, not done.
    orchestrator = Orchestrator(
        "pending", FakeLLM(settings=make_settings()), console=silent_console
    )
    orchestrator.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]

    assert orchestrator._is_done() is False


# ----------------------------------------------------------------- cycle budget


async def test_run_stops_after_exactly_max_cycles_cycles(
    make_settings, silent_console, stub_tester
) -> None:
    # max_cycles is a hard budget, and reaching it must not be reported as
    # success: the loop stops without claiming the goal is complete.
    settings = make_settings(max_cycles=2)
    stub_tester([failing_result("tests failed")])
    client = FakeLLM(
        {
            # The planner would happily plan forever, so only the budget stops it.
            str(settings.planner_model): ScriptedReplies([ONE_TASK, ONE_TASK, ONE_TASK]),
            str(settings.worker_model): WORKER_BOOM,
        },
        settings=settings,
    )
    orchestrator = Orchestrator("bounded", client, console=silent_console)

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert len(planner_prompts(client, str(settings.planner_model))) == 2
    assert orchestrator._is_done() is False
    assert "Reached cycle limit" in console_output(silent_console)


async def test_max_cycles_zero_runs_past_the_default_cap(
    make_settings, silent_console, stub_tester
) -> None:
    # max_cycles=0 means unlimited, not "stop at the default cap of three".
    #
    # Restructured: an always-stalling run can no longer demonstrate "unlimited",
    # because MAX_CONSECUTIVE_STALLED_CYCLES now aborts it at three -- which is
    # exactly what ``_max_cycles`` would otherwise do. The property under test is
    # that *nothing in the loop* stops it, so this run is scripted so that every
    # cycle after the first one makes progress (a completed task plus a green
    # suite), which resets the stall counter every time. The loop then has no
    # budget, no stall bound and no completion signal, and only the deliberate
    # cancellation can end it -- hence the assertion that five cycles really ran.
    settings = make_settings(max_cycles=0)
    # Cycle 1's tester run fails; every later run passes (the last entry repeats).
    stub_tester([failing_result("tests failed"), passing_result("all green")])
    client = FakeLLM(
        {
            # The first plan leaves a failed task in the accumulated history, and
            # that failure is what keeps _is_done() false for the whole run: with
            # every task completed and no feedback, cycle 2 would be "done".
            str(settings.planner_model): ScriptedReplies([ONE_TASK, OTHER_TASK]),
            # A worker reply with issues and no edits makes WorkerAgent.run raise,
            # so the first task fails and the later ones succeed.
            str(settings.worker_model): ScriptedReplies(
                [
                    worker_reply(issues=["could not apply the edit"]),
                    worker_reply(summary="wrote it"),
                ]
            ),
        },
        settings=settings,
        hook=CancelAfterPlannerCalls(str(settings.planner_model), after=5),
    )
    orchestrator = Orchestrator("unbounded", client, console=silent_console)

    # A cancellation must not be swallowed and retried as a failed cycle.
    with pytest.raises(asyncio.CancelledError):
        await orchestrator.run()

    assert orchestrator.cycles == 5, "a max_cycles=0 run was capped"
    assert orchestrator.cycles > 3
    # Progress was recognised every cycle after the first, so the stall bound
    # never fired -- which is the other half of "unlimited".
    assert orchestrator.consecutive_stalled_cycles == 0
    output = console_output(silent_console)
    assert "Reached cycle limit" not in output
    assert "no progress in the last" not in output


async def test_max_cycles_zero_still_completes_a_good_plan(
    make_settings, silent_console, stub_tester
) -> None:
    # An unlimited run is not a disabled run: the very first cycle that satisfies
    # _is_done() halts the loop on its own, with no cycle-limit message.
    #
    # Re-purposed to a *green* first cycle. It used to script a failing tester and
    # still assert "Goal complete", which is precisely the false pass that
    # ``_is_done()`` ignoring ``self.feedback`` used to produce: every task marked
    # "completed" while the suite was red, announced as success, exit code 0. The
    # failing-tester counterpart now lives in
    # ``test_max_cycles_zero_with_a_failing_tester_never_calls_the_goal_complete``.
    settings = make_settings(max_cycles=0)
    stub_tester([passing_result("all green")])
    client = FakeLLM(
        {
            str(settings.planner_model): ONE_TASK,
            str(settings.worker_model): worker_reply(summary="wrote it"),
        },
        settings=settings,
    )
    orchestrator = Orchestrator("unbounded but satisfied", client, console=silent_console)

    await orchestrator.run()

    assert orchestrator.cycles == 1
    assert orchestrator.tasks[0].status == "completed"
    assert "Goal complete" in console_output(silent_console)


async def test_max_cycles_zero_with_a_failing_tester_never_calls_the_goal_complete(
    make_settings, silent_console, stub_tester
) -> None:
    # The other half of the pair above: workers that all succeed and a tester that
    # keeps failing is NOT a finished goal. This used to print "Goal complete" and
    # exit 0 on a goal whose tests never passed, because _is_done() only looked at
    # task statuses and ignored the test feedback.
    #
    # With no cycle budget the only thing that can stop such a run is the stall
    # bound, so that is what must fire -- and it must say the goal is unverified.
    settings = make_settings(max_cycles=0)
    stub_tester([failing_result("tests failed")])
    client = FakeLLM(
        {
            str(settings.planner_model): ONE_TASK,
            str(settings.worker_model): worker_reply(summary="wrote it"),
        },
        settings=settings,
    )
    orchestrator = Orchestrator("unlimited but still red", client, console=silent_console)

    await orchestrator.run()

    output = console_output(silent_console)
    assert orchestrator.cycles == MAX_CONSECUTIVE_STALLED_CYCLES
    # Every task really did complete: only the red suite kept the run going.
    assert [task.status for task in orchestrator.tasks] == ["completed"] * orchestrator.cycles
    assert "Goal complete" not in output, "a red suite was reported as a success"
    assert "no progress in the last" in output
    assert "NOT verified" in output


async def test_run_aborts_after_the_consecutive_cycle_error_bound(
    make_settings, silent_console, stub_tester
) -> None:
    # A planner that only ever raises must not spin forever: run() counts
    # consecutive failed cycles and gives up at the bound.
    settings = make_settings(max_cycles=0)
    stub_tester([failing_result("tests failed")])
    client = FakeLLM(
        {str(settings.planner_model): PLANNER_BOOM},
        settings=settings,
    )
    orchestrator = Orchestrator("hopeless", client, console=silent_console)

    await orchestrator.run()

    assert orchestrator.cycles == MAX_CONSECUTIVE_CYCLE_ERRORS
    assert orchestrator.tasks == []
    output = console_output(silent_console)
    assert "consecutive cycle errors" in output


# ------------------------------------------------------------ dependency stages


def stage_ids(tasks: list[TaskModel]) -> list[list[str]]:
    """``Orchestrator._stage_tasks`` rendered as ids, which is the whole spec."""
    return [[task.id for task in stage] for stage in Orchestrator._stage_tasks(tasks)]


def test_stage_tasks_puts_a_dependent_in_a_later_stage() -> None:
    # A task may only run once everything it depends on is already scheduled.
    a = TaskModel(id="a", description="a")
    b = TaskModel(id="b", description="b", dependencies=["a"])

    assert stage_ids([a, b]) == [["a"], ["b"]]


def test_stage_tasks_groups_independent_tasks_into_one_stage() -> None:
    # Nothing to order them by, so everything runs concurrently in one batch.
    tasks = [TaskModel(id=task_id, description=task_id) for task_id in ("a", "b", "c")]

    assert stage_ids(tasks) == [["a", "b", "c"]]


def test_stage_tasks_treats_unknown_dependencies_as_satisfied() -> None:
    # A dependency on an id that is not in the plan can never be satisfied, so it
    # must not deadlock the task: it is schedulable in the first stage.
    a = TaskModel(id="a", description="a", dependencies=["not-in-plan"])

    assert stage_ids([a]) == [["a"]]


def test_stage_tasks_flushes_a_pure_dependency_cycle_without_hanging() -> None:
    # a needs b and b needs a. The scheduler cannot order them, so it flushes
    # the remainder as one stage instead of spinning forever.
    a = TaskModel(id="a", description="a", dependencies=["b"])
    b = TaskModel(id="b", description="b", dependencies=["a"])

    assert stage_ids([a, b]) == [["a", "b"]]


def test_stage_tasks_runs_independent_tasks_before_a_dependency_cycle() -> None:
    # c is independent, so it goes first; the unorderable pair is flushed last
    # rather than holding up work that could have run already.
    a = TaskModel(id="a", description="a", dependencies=["b"])
    b = TaskModel(id="b", description="b", dependencies=["a"])
    c = TaskModel(id="c", description="c")

    assert stage_ids([a, b, c]) == [["c"], ["a", "b"]]


async def test_run_never_executes_a_dependent_task_before_its_dependency(
    make_settings, silent_console, stub_tester
) -> None:
    # The staging above has to survive the whole cycle: task 2 declares
    # dependencies ["1"] and must be handed to a worker only after task 1 is done.
    settings = make_settings()
    stub_tester([failing_result("tests failed")])
    executed: list[str] = []

    async def hook(*, model: str, prompt: str, system: str) -> None:
        if model == settings.worker_model:
            executed.append(task_id_in_prompt(prompt))

    client = FakeLLM(
        {
            str(settings.planner_model): plan_reply(
                {"id": "1", "description": "first", "files": []},
                {"id": "2", "description": "second", "files": [], "dependencies": ["1"]},
            ),
            str(settings.worker_model): worker_reply(summary="wrote it"),
        },
        settings=settings,
        hook=hook,
    )
    orchestrator = Orchestrator("ordered", client, console=silent_console)

    await orchestrator.run()

    # The exact call list is deliberately NOT asserted as ["1", "2"]: while the
    # suite is red the run legitimately retries, so the loop legitimately replans
    # the same two tasks and `executed` grows to ['1','2','1','2',...]. What must
    # hold every time is the ordering invariant -- and proving the list repeats
    # whole rounds is itself the proof that more than one cycle ran.
    assert len(executed) % 2 == 0
    assert executed == ["1", "2"] * (len(executed) // 2)
    assert len(executed) >= 4, "more than one cycle must have run for this to mean anything"
    assert executed.index("2") > executed.index("1"), "the dependent task ran too early"
    assert [task.status for task in orchestrator.tasks] == ["completed"] * len(orchestrator.tasks)


async def test_run_never_exceeds_max_parallel_tasks(
    make_settings, silent_console, stub_tester
) -> None:
    # Five independent tasks with a cap of two: the semaphore in _run_task, not
    # the scheduler's luck, is what keeps in-flight workers at two. The sleep
    # only widens the overlap window; the assertion is on the counter.
    settings = make_settings(max_parallel_tasks=2, max_cycles=1)
    stub_tester([failing_result("tests failed")])
    in_flight = 0
    peak = 0

    async def hook(*, model: str, prompt: str, system: str) -> None:
        nonlocal in_flight, peak
        if model != settings.worker_model:
            return
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1

    client = FakeLLM(
        {str(settings.worker_model): worker_reply(summary="wrote it")},
        settings=settings,
        hook=hook,
    )
    orchestrator = Orchestrator("throttled", client, console=silent_console)
    # Injected rather than planned: PlannerAgent caps a plan at
    # max_parallel_tasks, which would silently reduce this to a two-task run.
    orchestrator.planner = FixedPlanner(
        Plan(
            tasks=[TaskModel(id=str(index), description=f"task {index}") for index in range(1, 6)],
            rationale="five independent tasks",
        )
    )

    await orchestrator.run()

    assert peak == 2, f"expected exactly two concurrent workers, saw {peak}"
    assert len([task for task in orchestrator.tasks if task.status == "completed"]) == 5


# ----------------------------------------------------------------- goal & tests


async def test_run_keeps_the_original_goal_and_feeds_failures_back_to_the_planner(
    make_settings, silent_console, stub_tester
) -> None:
    # The old code overwrote `goal` with the failure text, so the second cycle
    # planned a fix for "boom" and the user's actual goal was gone for good.
    settings = make_settings(max_cycles=2)
    failure = "boom: test_retries_back_off failed"
    stub_tester([failing_result(failure)])
    client = FakeLLM(
        {
            str(settings.planner_model): ScriptedReplies([ONE_TASK, OTHER_TASK]),
            # The worker keeps failing so the loop reaches a second cycle.
            str(settings.worker_model): WORKER_BOOM,
        },
        settings=settings,
    )
    orchestrator = Orchestrator("ship the retry logic", client, console=silent_console)

    await orchestrator.run()

    assert orchestrator.original_goal == "ship the retry logic"
    assert orchestrator.goal == orchestrator.original_goal, "the goal was overwritten"
    assert orchestrator.feedback == failure

    # The next planner prompt must carry the goal *and* the feedback, not
    # either one alone.
    goal_input = orchestrator._goal_input()
    assert "ship the retry logic" in goal_input
    assert failure in goal_input
    second_prompt = planner_prompts(client, str(settings.planner_model))[1]
    assert "ship the retry logic" in second_prompt
    assert failure in second_prompt
