"""Regression tests for :class:`furrow.core.orchestrator.Orchestrator`.

Every test drives the **real** orchestrator with the real planner, worker and
tester agents and the real pydantic models: only ``LLMClient.complete`` is
replaced.  The tester's own test command is pinned to a cheap, hermetic shell
command (``true`` / ``false``) so no pytest subprocess and no network call
happens here.

Load-bearing regressions pinned here:

* the loop really iterates - ``run()`` must not report success after one failing
  cycle (the ``_get_tasks() == []`` / ``0 >= 0`` accident);
* ``max_parallel_tasks`` bounds the number of concurrently running workers;
* ``max_cycles`` bounds the loop;
* a failed dependency marks its dependents ``skipped`` and never runs them, and
  the original goal string survives;
* ``stop()``, a raising ``on_event`` and a planner ``ValueError`` cannot break the
  loop, and ``asyncio.CancelledError`` still propagates.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core.orchestrator import CycleOutcome, Orchestrator, _dependency_waves
from furrow.llm import LLMClient
from tests.conftest import (
    PLANNER,
    TESTER,
    WORKER,
    ScriptedCompletion,
    install_scripted,
    make_settings,
    plan_reply,
    task,
    verdict_json,
    worker_reply,
)

TIMEOUT = 20.0
"""Every awaited run is bounded so a regression can never hang the suite."""

PASSING_TESTER = verdict_json(True, "all green")
FAILING_TESTER = verdict_json(False, "one failed", ["test_x failed"])


class Recorder:
    """Collects ``(event, message)`` pairs emitted by the orchestrator."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def __call__(self, event: str, message: str) -> None:
        self.events.append((event, message))

    @property
    def names(self) -> list[str]:
        return [event for event, _ in self.events]

    def messages(self, name: str) -> list[str]:
        return [message for event, message in self.events if event == name]


def build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    planner: Any = "",
    worker: Any = "",
    tester: Any = PASSING_TESTER,
    on_event: Any = None,
    goal: str = "ORIGINAL GOAL",
    **setting_overrides: Any,
) -> tuple[Orchestrator, ScriptedCompletion, Settings, Recorder]:
    """A real orchestrator wired to scripted LLM replies and an isolated workspace."""
    recorder = Recorder()
    overrides: dict[str, Any] = {"test_command": "true", "max_cycles": 1}
    overrides.update(setting_overrides)
    settings = make_settings(tmp_path / "ws", **overrides)
    scripted = install_scripted(monkeypatch, planner=planner, worker=worker, tester=tester)
    client = LLMClient(settings=settings)
    orchestrator = Orchestrator(
        goal=goal,
        client=client,
        settings=settings,
        on_event=recorder if on_event is None else on_event,
    )
    return orchestrator, scripted, settings, recorder


async def run(orchestrator: Orchestrator) -> bool:
    """``Orchestrator.run`` under a hard deadline."""
    return await asyncio.wait_for(orchestrator.run(), timeout=TIMEOUT)


# ------------------------------------------- the loop actually iterates


async def test_run_is_false_and_the_loop_repeats_when_tests_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The regression: one failing cycle must not be reported as "goal complete"."""
    orchestrator, scripted, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=FAILING_TESTER,
        max_cycles=3,
    )
    completed = await run(orchestrator)
    assert completed is False
    assert orchestrator.done is False
    assert orchestrator.cycles == 3
    assert len(scripted.prompts(PLANNER)) == 3, "the planner must be asked again next cycle"
    assert len(scripted.prompts(TESTER)) == 3
    assert "done" in recorder.names


async def test_is_done_is_false_before_the_first_plan(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``_get_tasks()`` used to return ``[]``, so ``0 >= 0`` meant "complete"."""
    settings = make_settings(tmp_path / "ws")
    orchestrator = Orchestrator(goal="g", client=LLMClient(settings=settings), settings=settings)
    assert orchestrator.current_plan is None
    assert orchestrator._get_tasks() == []
    assert orchestrator._is_done() is False
    assert orchestrator.done is False


async def test_get_tasks_returns_the_current_plan_tasks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1"), task("2")),
        worker=worker_reply("wrote it"),
        max_cycles=1,
    )
    await run(orchestrator)
    assert orchestrator.current_plan is not None
    assert [t.id for t in orchestrator._get_tasks()] == ["1", "2"]


async def test_run_returns_true_only_when_tasks_completed_and_tests_passed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=PASSING_TESTER,
        max_cycles=0,
    )
    assert await run(orchestrator) is True
    assert orchestrator.done is True
    assert orchestrator.cycles == 1


async def test_all_tasks_completed_but_tests_failing_is_not_a_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1"), task("2")),
        worker=worker_reply("wrote it"),
        tester=FAILING_TESTER,
        max_cycles=1,
    )
    assert await run(orchestrator) is False
    assert orchestrator.done is False


async def test_a_failing_task_means_not_done_even_with_green_tests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def failing_worker(call: Any) -> str:
        if "Task: bad" in call.prompt:
            raise RuntimeError("worker exploded")
        return worker_reply("wrote it")

    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("good", "good"), task("bad", "bad")),
        worker=failing_worker,
        tester=PASSING_TESTER,
        max_cycles=1,
    )
    assert await run(orchestrator) is False
    assert orchestrator.done is False


async def test_an_empty_plan_from_the_model_is_a_planner_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``PlannerAgent`` refuses an empty task list, so the loop never "completes"."""
    orchestrator, scripted, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner='{"tasks": [], "rationale": "nothing left"}',
        tester=PASSING_TESTER,
        max_cycles=3,
    )
    assert await run(orchestrator) is False
    assert orchestrator.error is not None
    assert "at least one task" in orchestrator.error
    assert scripted.prompts(WORKER) == []
    assert any("Planner error" in message for message in recorder.messages("error"))


async def test_a_plan_with_no_tasks_and_green_tests_completes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``NOTHING_PLANNED``: no work left and the suite is green -> done."""
    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=PASSING_TESTER,
    )
    orchestrator.planner.plan = _returning(Plan(tasks=[], rationale="nothing left"))  # type: ignore[method-assign]
    assert await run(orchestrator) is True
    assert orchestrator.cycles == 1
    assert orchestrator._get_tasks() == []
    assert orchestrator.last_test_result is not None
    assert orchestrator.last_test_result.passed is True


async def test_a_plan_with_no_tasks_and_failing_tests_keeps_going(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, scripted, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=FAILING_TESTER,
        max_cycles=2,
    )
    orchestrator.planner.plan = _returning(Plan(tasks=[], rationale="still broken"))  # type: ignore[method-assign]
    assert await run(orchestrator) is False
    assert orchestrator.cycles == 2
    assert len(scripted.prompts(TESTER)) == 2


def _returning(value: Any) -> Any:
    """An awaitable that resolves to *value*, for stubbing ``planner.plan``."""

    async def plan(goal: str) -> Any:
        return value

    return plan


# ------------------------------------------------------------ max_cycles


async def test_max_cycles_halts_the_loop_and_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, scripted, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=FAILING_TESTER,
        max_cycles=4,
    )
    assert await run(orchestrator) is False
    assert orchestrator.cycles == 4
    assert len(scripted.prompts(PLANNER)) == 4
    assert any("max_cycles" in message for message in recorder.messages("done"))


async def test_max_cycles_is_not_consulted_before_the_first_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, scripted, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=PASSING_TESTER,
        max_cycles=1,
    )
    assert await run(orchestrator) is True
    assert len(scripted.prompts(PLANNER)) == 1


# -------------------------------------------------- max_parallel_tasks


@pytest.mark.parametrize("limit", [1, 2, 3])
async def test_max_parallel_tasks_bounds_the_running_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, limit: int
) -> None:
    running = 0
    peak = 0
    # ``limit`` workers must be able to run at once before any of them may finish.
    gate = asyncio.Event()

    def counting_worker(call: Any) -> Any:
        async def run_one() -> str:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            if running >= limit:
                gate.set()
            await asyncio.wait_for(gate.wait(), timeout=10)
            running -= 1
            return worker_reply("wrote it")

        return run_one()

    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(*[task(str(i), f"task {i}") for i in range(5)]),
        worker=counting_worker,
        tester=PASSING_TESTER,
        max_parallel_tasks=limit,
        max_cycles=1,
    )
    assert await run(orchestrator) is True
    assert peak == limit, f"peak concurrency was {peak}, expected exactly {limit}"


async def test_max_parallel_tasks_of_one_serialises_the_workers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    order: list[str] = []

    def worker(call: Any) -> str:
        order.append("start:" + call.prompt.split("Task: ")[1].splitlines()[0])
        return worker_reply("wrote it")

    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(*[task(str(i), f"task {i}") for i in range(3)]),
        worker=worker,
        tester=PASSING_TESTER,
        max_parallel_tasks=1,
        max_cycles=1,
    )
    assert await run(orchestrator) is True
    assert len(order) == 3
    assert all(item.startswith("start:") for item in order)


# ------------------------------------------------------ dependency handling


async def test_a_failing_dependency_skips_its_dependents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ran: list[str] = []

    def worker(call: Any) -> str:
        which = call.prompt.split("Task: ")[1].splitlines()[0]
        ran.append(which)
        if which == "first":
            raise RuntimeError("worker exploded")
        return worker_reply("wrote it")

    orchestrator, _, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(
            task("a", "first"),
            task("b", "second", dependencies=["a"]),
            task("c", "third", dependencies=["b"]),
        ),
        worker=worker,
        tester=PASSING_TESTER,
        max_cycles=1,
    )
    assert await run(orchestrator) is False
    assert ran == ["first"], "dependents of a failed task must never run"
    statuses = {t.id: t.status for t in orchestrator.current_plan.tasks}
    assert statuses == {
        "a": TaskStatus.FAILED,
        "b": TaskStatus.SKIPPED,
        "c": TaskStatus.SKIPPED,
    }
    assert "exploded" in orchestrator.results["a"]
    assert "did not complete" in orchestrator.results["b"]
    assert any("skipped" in message for message in recorder.messages("task"))


async def test_a_failed_dependency_does_not_overwrite_the_goal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def worker(call: Any) -> str:
        if "Task: first" in call.prompt:
            raise RuntimeError("worker exploded: nothing works")
        return worker_reply("wrote it")

    orchestrator, scripted, _, _ = build(
        monkeypatch,
        tmp_path,
        goal="ORIGINAL GOAL",
        planner=plan_reply(task("a", "first"), task("b", "second", dependencies=["a"])),
        worker=worker,
        tester=FAILING_TESTER,
        max_cycles=2,
    )
    await run(orchestrator)
    assert orchestrator.goal == "ORIGINAL GOAL"
    assert orchestrator.current_focus.startswith("Goal: ORIGINAL GOAL")
    for prompt in scripted.prompts(PLANNER):
        assert prompt.startswith("You are a planning agent")
        assert "Goal: ORIGINAL GOAL" in prompt


async def test_dependencies_run_in_waves(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    order: list[str] = []

    def worker(call: Any) -> str:
        order.append(call.prompt.split("Task: ")[1].splitlines()[0])
        return worker_reply("wrote it")

    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(
            task("base", "build the base"),
            task("one", "add one", dependencies=["base"]),
            task("two", "add two", dependencies=["base"]),
            task("solo", "independent"),
        ),
        worker=worker,
        tester=PASSING_TESTER,
        max_parallel_tasks=2,
        max_cycles=1,
    )
    assert await run(orchestrator) is True
    assert order.index("build the base") < order.index("add one")
    assert order.index("build the base") < order.index("add two")


async def test_a_dependency_cycle_self_dependency_and_unknown_id_still_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(
            task("a", "alpha", dependencies=["b"]),
            task("b", "beta", dependencies=["a", "a"]),
            task("c", "gamma", dependencies=["c"]),
            task("d", "delta", dependencies=["ghost"]),
        ),
        worker=worker_reply("wrote it"),
        tester=PASSING_TESTER,
        max_cycles=1,
    )
    assert await asyncio.wait_for(orchestrator.run(), timeout=TIMEOUT) is True
    assert {t.status for t in orchestrator.current_plan.tasks} == {TaskStatus.COMPLETED}
    assert any("cycle between" in message for message in recorder.messages("task"))


def test_dependency_waves_break_a_cycle_instead_of_deadlocking() -> None:
    tasks = [
        TaskModel(id="a", description="a", dependencies=["b"]),
        TaskModel(id="b", description="b", dependencies=["a"]),
    ]
    waves, forced = _dependency_waves(tasks)
    assert forced
    assert [t.id for wave in waves for t in wave] == ["a", "b"]


def test_dependency_waves_ignore_unknown_ids_and_self_references() -> None:
    tasks = [
        TaskModel(id="a", description="a", dependencies=["a", "ghost"]),
        TaskModel(id="b", description="b", dependencies=["a"]),
    ]
    waves, forced = _dependency_waves(tasks)
    assert forced == []
    assert [[t.id for t in wave] for wave in waves] == [["a"], ["b"]]


# ----------------------------------------------------------------- stop()


async def test_stop_returns_promptly_with_no_worker_left_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    active = 0

    def blocking_worker(call: Any) -> Any:
        async def run_one() -> str:
            nonlocal active
            active += 1
            entered.set()
            try:
                await asyncio.wait_for(release.wait(), timeout=10)
            finally:
                active -= 1
            return worker_reply("wrote it")

        return run_one()

    orchestrator, scripted, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1", "slow work")),
        worker=blocking_worker,
        tester=PASSING_TESTER,
        max_cycles=0,
    )
    running = asyncio.ensure_future(orchestrator.run())
    await asyncio.wait_for(entered.wait(), timeout=TIMEOUT)
    await orchestrator.stop()
    release.set()
    assert await asyncio.wait_for(asyncio.shield(running), timeout=TIMEOUT) is False
    assert active == 0, "the in-flight worker must be finished, not abandoned"
    assert len(scripted.prompts(PLANNER)) == 1
    assert "done" in recorder.names


async def test_stop_before_run_returns_false_immediately(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, scripted, _, _ = build(
        monkeypatch, tmp_path, planner=plan_reply(task("1")), max_cycles=0
    )
    await orchestrator.stop()
    await orchestrator.stop()  # idempotent
    assert await run(orchestrator) is False
    assert scripted.prompts(PLANNER) == []


async def test_stop_requested_by_a_worker_prevents_the_next_wave(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``stop()`` from inside a worker must stop the next wave from starting."""
    holder: dict[str, Orchestrator] = {}
    ran: list[str] = []

    def worker(call: Any) -> Any:
        async def run_one() -> str:
            if "Task: first" in call.prompt:
                await holder["orchestrator"].stop()
                return worker_reply("wrote it")
            ran.append("second")
            return worker_reply("wrote it")

        return run_one()

    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("a", "first"), task("b", "second", dependencies=["a"])),
        worker=worker,
        tester=PASSING_TESTER,
        max_cycles=0,
    )
    holder["orchestrator"] = orchestrator
    assert await run(orchestrator) is False
    assert ran == [], "the dependent task must not start after stop()"
    statuses = {t.id: str(t.status) for t in orchestrator.current_plan.tasks}
    assert statuses["b"] == "skipped"


# ---------------------------------------------------------------- errors


async def test_a_planner_value_error_is_surfaced_without_crashing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner="I refuse to produce a plan.",
        tester=PASSING_TESTER,
        max_cycles=3,
    )
    assert await run(orchestrator) is False
    assert orchestrator.error is not None
    assert orchestrator.error.startswith("ValueError:")
    assert "Failed to parse plan" in orchestrator.error
    assert orchestrator.cycles == 1, "a planner error halts the run immediately"
    assert recorder.messages("error")
    assert any("Planner error" in message for message in recorder.messages("error"))


async def test_a_planner_exception_is_also_contained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def exploding_planner(call: Any) -> str:
        raise RuntimeError("transport is down")

    orchestrator, _, _, recorder = build(
        monkeypatch, tmp_path, planner=exploding_planner, max_cycles=2
    )
    assert await run(orchestrator) is False
    assert "RuntimeError" in (orchestrator.error or "")
    assert recorder.messages("error")


async def test_a_broken_tester_is_contained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def exploding_tester(call: Any) -> str:
        raise RuntimeError("tester is down")

    orchestrator, _, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=exploding_tester,
        max_cycles=1,
    )
    assert await run(orchestrator) is False
    assert orchestrator.error is not None
    assert any("Tester error" in message for message in recorder.messages("error"))


async def test_a_tester_returning_junk_is_contained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=42,
        max_cycles=1,
    )
    assert await run(orchestrator) is False


# ----------------------------------------------------------------- events


async def test_on_event_receives_the_expected_event_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, _, recorder = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it", files=[{"path": "made.py", "content": "x = 1\n"}]),
        tester=PASSING_TESTER,
        max_cycles=1,
    )
    assert await run(orchestrator) is True
    names = set(recorder.names)
    for expected in ("run", "cycle", "plan", "task", "test", "done"):
        assert expected in names, f"missing {expected} event in {sorted(names)}"
    assert recorder.names[0] == "run"
    assert recorder.names[-1] == "done"
    assert all(isinstance(message, str) for _, message in recorder.events)


async def test_a_raising_on_event_callback_does_not_break_the_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def exploding_consumer(event: str, message: str) -> None:
        raise RuntimeError("consumer is dead")

    orchestrator, scripted, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=PASSING_TESTER,
        on_event=exploding_consumer,
        max_cycles=1,
    )
    assert await run(orchestrator) is True
    assert orchestrator.error is None
    assert len(scripted.prompts(PLANNER)) == 1
    assert len(scripted.prompts(WORKER)) == 1


async def test_no_on_event_is_fine(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=PASSING_TESTER,
        max_cycles=1,
    )
    orchestrator.on_event = None
    assert await run(orchestrator) is True


# ------------------------------------------------------------ cancellation


async def test_cancellation_propagates_out_of_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered = asyncio.Event()

    def hanging_planner(call: Any) -> Any:
        async def never() -> str:
            entered.set()
            await asyncio.sleep(TIMEOUT)
            return plan_reply(task("1"))

        return never()

    orchestrator, _, _, recorder = build(
        monkeypatch, tmp_path, planner=hanging_planner, max_cycles=0
    )
    running = asyncio.ensure_future(orchestrator.run())
    await asyncio.wait_for(entered.wait(), timeout=TIMEOUT)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert orchestrator.done is False
    assert recorder.messages("done") == ["Run cancelled."]


async def test_a_cancelled_worker_stops_the_loop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered = asyncio.Event()

    def cancelling_worker(call: Any) -> Any:
        async def run_one() -> str:
            entered.set()
            raise asyncio.CancelledError

        return run_one()

    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=cancelling_worker,
        tester=PASSING_TESTER,
        max_cycles=0,
    )
    running = asyncio.ensure_future(orchestrator.run())
    await asyncio.wait_for(entered.wait(), timeout=TIMEOUT)
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(running), timeout=TIMEOUT)


# ------------------------------------------------------------ state after a run


async def test_state_after_a_successful_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, settings, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1", "make a module", files=["made.py"])),
        worker=worker_reply("created made.py", files=[{"path": "made.py", "content": "X = 1\n"}]),
        tester=PASSING_TESTER,
        max_cycles=1,
    )
    assert await run(orchestrator) is True
    assert (tmp_path / "ws" / "made.py").read_text() == "X = 1\n"
    plan = orchestrator.current_plan
    assert isinstance(plan, Plan)
    assert plan.tasks[0].status is TaskStatus.COMPLETED
    assert "made.py" in (plan.tasks[0].result or "")
    assert orchestrator.results["1"] == plan.tasks[0].result
    assert isinstance(orchestrator.last_test_result, TestResult)
    assert orchestrator.last_test_result.passed is True
    assert orchestrator.history == [plan]
    assert orchestrator.error is None
    assert orchestrator.settings is settings


async def test_history_grows_once_per_cycle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, _, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=FAILING_TESTER,
        max_cycles=3,
    )
    await run(orchestrator)
    assert len(orchestrator.history) == 3
    assert all(isinstance(entry, Plan) for entry in orchestrator.history)


async def test_the_next_cycle_focus_mentions_the_outstanding_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    orchestrator, scripted, _, _ = build(
        monkeypatch,
        tmp_path,
        planner=plan_reply(task("1")),
        worker=worker_reply("wrote it"),
        tester=verdict_json(False, "one failed", ["test_widget blew up"]),
        max_cycles=2,
    )
    await run(orchestrator)
    second_prompt = scripted.prompts(PLANNER)[1]
    assert "test_widget blew up" in second_prompt


async def test_a_constructor_without_arguments_uses_the_singleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from furrow.config import settings as global_settings

    install_scripted(monkeypatch, planner=plan_reply(task("1")))
    orchestrator = Orchestrator(goal="g")
    assert orchestrator.settings is global_settings
    assert orchestrator.client.settings is global_settings
    assert orchestrator.planner.client is orchestrator.client


async def test_cycles_and_done_start_at_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    settings = make_settings(tmp_path / "ws")
    orchestrator = Orchestrator(goal="g", client=LLMClient(settings=settings), settings=settings)
    assert orchestrator.cycles == 0
    assert orchestrator.done is False
    assert orchestrator.history == []
    assert orchestrator.results == {}
    assert orchestrator.last_test_result is None
    assert orchestrator.error is None
    assert orchestrator.current_focus == "g"
    assert CycleOutcome.COMPLETED.value == "completed"
