"""Tests for the plan -> execute -> test loop in :mod:`furrow.core.orchestrator`.

Every test drives the real ``Orchestrator`` (and the real planner/worker/tester
agents) with a fake LLM client and a scripted test runner, so no subprocess, no
network and no API key is ever touched.  Anything that could hang is wrapped in
``asyncio.wait_for``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from typing import Any

import aiofiles
import pytest
from rich.console import Console

from furrow.agents import tester as tester_mod
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TestRun as RunOutcome
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel, TaskStatus
from furrow.core.events import Event, EventBus
from furrow.core.orchestrator import Orchestrator
from furrow.state import RunState, StateStore

GOAL = "build the thing"
PLAN_JSON: dict[str, Any] = {
    "tasks": [{"id": "t1", "description": "write the module"}],
    "rationale": "one step is enough",
}
PASS_OUTPUT = "1 passed in 0.01s"
FAIL_OUTPUT = (
    "FAILED tests/test_widget.py::test_rolls - AssertionError: 1 != 2\n"
    "ERROR tests/test_widget.py::test_fails - RuntimeError: nope\n"
)
PASS: list[dict[str, Any]] = [{"passed": True}]
FAIL: list[dict[str, Any]] = [{"passed": False}]
#: Two distinct failure signatures: alternating them never repeats a signature.
FAIL_A: dict[str, Any] = {
    "passed": False,
    "output": "FAILED tests/test_a.py::test_a - AssertionError: first",
}
FAIL_B: dict[str, Any] = {
    "passed": False,
    "output": "FAILED tests/test_b.py::test_b - AssertionError: second",
}
ALTERNATE_FAIL: list[dict[str, Any]] = [FAIL_A, FAIL_B, FAIL_A, FAIL_B]


class FakeClient:
    """The slice of :class:`furrow.llm.LLMClient` that the loop actually calls."""

    def __init__(
        self,
        settings: Settings,
        *,
        plans: list[dict[str, Any]] | None = None,
        test_verdict: dict[str, Any] | None = None,
        worker_result: str = "worker finished the task",
        json_error: BaseException | None = None,
        worker_script: list[BaseException | None] | None = None,
        delay: float = 0.0,
    ) -> None:
        self.settings = settings
        self._plans = list(plans) if plans is not None else [PLAN_JSON]
        self._plan_calls = 0
        self._worker_calls = 0
        self._worker_script = list(worker_script or [])
        self.test_verdict = dict(test_verdict or {})
        self.worker_result = worker_result
        self.json_error = json_error
        self.delay = delay
        self.closed = False
        self.prompts: list[tuple[str, str | None]] = []

    async def complete_json(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        *,
        repair_attempts: int = 1,
    ) -> Any:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.prompts.append((prompt, model))
        if self.json_error is not None:
            raise self.json_error
        if model == self.settings.tester_model:
            return dict(self.test_verdict)
        index = min(self._plan_calls, len(self._plans) - 1)
        self._plan_calls += 1
        return self._plans[index]

    async def complete(
        self,
        prompt: str,
        system: str = "",
        model: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self.delay:
            await asyncio.sleep(self.delay)
        self.prompts.append((prompt, model))
        index = min(self._worker_calls, len(self._worker_script) - 1)
        self._worker_calls += 1
        if self._worker_script:
            error = self._worker_script[index]
            if error is not None:
                raise error
        return self.worker_result

    async def read_file(self, path: str | Path) -> str:
        async with aiofiles.open(path) as handle:
            return await handle.read()

    def list_files(self, directory: str | Path) -> list[str]:
        root = Path(directory)
        return [path.name for path in sorted(root.glob("*")) if path.is_file()]

    async def close(self) -> None:
        self.closed = True


class RecordingStore(StateStore):
    """A :class:`StateStore` that remembers how often it was written."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.saves = 0

    def save(self, state: RunState) -> None:
        self.saves += 1
        super().save(state)


def patch_tester(monkeypatch: pytest.MonkeyPatch, verdicts: list[dict[str, Any]]) -> None:
    """Script ``TesterAgent.run_tests`` so the real test runner never spawns.

    The last entry repeats once the script runs out; the parsing, summary and
    failure-extraction code of the real tester agent still runs.
    """
    script = [dict(item) for item in verdicts]
    tail = dict(script[-1]) if script else {"passed": True}

    async def run_tests(self: tester_mod.TesterAgent) -> RunOutcome:
        item = script.pop(0) if script else tail
        passed = bool(item.get("passed", True))
        return RunOutcome(
            command=["python", "-m", "pytest", "-q"],
            returncode=0 if passed else 1,
            output=str(item.get("output") or (PASS_OUTPUT if passed else FAIL_OUTPUT)),
            found=True,
        )

    monkeypatch.setattr(tester_mod.TesterAgent, "run_tests", run_tests)


def build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    verdicts: list[dict[str, Any]] = PASS,
    plans: list[dict[str, Any]] | None = None,
    test_verdict: dict[str, Any] | None = None,
    json_error: BaseException | None = None,
    worker_script: list[BaseException | None] | None = None,
    delay: float = 0.0,
    **settings_kwargs: Any,
) -> tuple[Orchestrator, FakeClient, asyncio.Queue[Event]]:
    """Assemble an orchestrator wired to stubs, plus its client and event queue."""
    settings = Settings(
        _env_file=None,
        workspace=tmp_path,
        state_path=tmp_path / "state.json",
        **settings_kwargs,
    )
    patch_tester(monkeypatch, verdicts)
    bus = EventBus()
    client = FakeClient(
        settings,
        plans=plans,
        test_verdict=test_verdict,
        json_error=json_error,
        worker_script=worker_script,
        delay=delay,
    )
    orchestrator = Orchestrator(
        goal=GOAL,
        client=client,
        settings=settings,
        store=RecordingStore(settings.resolved_state_path),
        events=bus,
        console=Console(quiet=True),
    )
    return orchestrator, client, bus.subscribe()


def drain(queue: asyncio.Queue[Event]) -> list[Event]:
    """Take every event a subscriber has collected so far."""
    events: list[Event] = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


async def run_for(orchestrator: Orchestrator, timeout: float = 15.0) -> RunState:
    return await asyncio.wait_for(orchestrator.run(), timeout=timeout)


# --------------------------------------------------------------------- the loop


async def test_happy_path_finishes_in_a_single_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(tmp_path, monkeypatch, max_cycles=0)

    state = await run_for(orchestrator)

    assert state.done is True
    assert state.halted_reason is None
    assert state.goal == GOAL
    assert state.cycles == 1
    assert orchestrator.cycles == 1
    assert orchestrator.is_running is False
    assert [task.status for task in state.tasks] == [TaskStatus.COMPLETED]
    assert state.tasks[0].result == "worker finished the task"
    assert state.fix_instructions == ""
    assert len(state.history) == 1
    assert state.history[0]["passed"] is True
    # "It does not spin": one cycle is enough, the loop stops immediately.
    assert [event.kind for event in drain(queue)].count("cycle_start") == 1


async def test_goal_survives_failing_cycles_and_failures_accumulate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, client, _ = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=3,
        stagnation_limit=0,
    )

    state = await run_for(orchestrator)

    # The old loop overwrote the goal with the failure text on every cycle.
    assert state.goal == GOAL
    assert orchestrator.goal == GOAL
    assert state.fix_instructions == orchestrator.fix_instructions
    assert "FAILED tests/test_widget.py::test_rolls" in state.fix_instructions
    assert "ERROR tests/test_widget.py::test_fails" in state.fix_instructions
    # The failures reach the planner as context, never as the objective.
    planner_prompts = [
        prompt for prompt, model in client.prompts if model == orchestrator.settings.planner_model
    ]
    assert len(planner_prompts) == 3
    assert all(f"Goal: {GOAL}" in prompt for prompt in planner_prompts)
    assert "Outstanding failures to fix" in planner_prompts[-1]
    assert "FAILED tests/test_widget.py::test_rolls" in planner_prompts[-1]
    assert state.done is False
    assert state.halted_reason == "max_cycles reached"


async def test_fix_instructions_are_deduplicated_across_cycles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, _ = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=2,
        stagnation_limit=0,
    )

    state = await run_for(orchestrator)

    lines = state.fix_instructions.splitlines()
    assert len(lines) == len(set(lines))
    assert len(lines) == 2


async def test_max_cycles_is_honored_even_when_tests_never_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=3,
        stagnation_limit=0,
    )

    state = await run_for(orchestrator)

    assert state.cycles == 3
    assert state.halted_reason == "max_cycles reached"
    assert state.done is False
    assert len(state.history) == 3
    assert [entry["cycle"] for entry in state.history] == [1, 2, 3]
    assert [event.kind for event in drain(queue)].count("cycle_start") == 3


async def test_stagnation_guard_halts_on_repeated_identical_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, _ = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=0,  # unbounded: only the stagnation guard may end this run
        stagnation_limit=2,
    )

    state = await run_for(orchestrator)

    assert state.halted_reason == "no progress on repeated test failures"
    assert state.cycles == 2
    assert state.done is False


async def test_stagnation_guard_only_counts_repeated_signatures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, _ = build(
        tmp_path,
        monkeypatch,
        verdicts=ALTERNATE_FAIL,
        max_cycles=4,
        stagnation_limit=2,
    )

    state = await run_for(orchestrator)

    assert state.halted_reason == "max_cycles reached"
    assert state.cycles == 4


async def test_stagnation_limit_zero_disables_the_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, _ = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=3,
        stagnation_limit=0,
    )

    state = await run_for(orchestrator)

    assert state.halted_reason == "max_cycles reached"
    assert state.cycles == 3


async def test_consecutive_error_budget_stops_the_loop_without_an_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(
        tmp_path,
        monkeypatch,
        json_error=RuntimeError("ANTHROPIC_API_KEY is not set"),
        max_cycles=0,
        max_consecutive_errors=3,
    )

    state = await run_for(orchestrator)

    assert state.halted_reason == (
        "too many consecutive errors (3); check the LLM provider configuration"
    )
    assert state.cycles == 3
    assert state.done is False
    # Every cycle aborted before the tests ran, and that is recorded, not lost.
    assert [entry["cycle"] for entry in state.history] == [1, 2, 3]
    assert all(entry["passed"] is False for entry in state.history)
    assert all(
        entry["summary"] == "cycle aborted before tests ran" for entry in state.history
    )
    events = drain(queue)
    assert [event.kind for event in events].count("error") == 3
    assert all(
        "ANTHROPIC_API_KEY" in event.message for event in events if event.kind == "error"
    )


async def test_a_healthy_cycle_resets_the_consecutive_error_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    boom = RuntimeError("worker exploded")
    orchestrator, _, _ = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        worker_script=[boom, boom, None],
        max_cycles=4,
        max_consecutive_errors=2,
        stagnation_limit=0,
    )

    state = await run_for(orchestrator)

    # Cycle 1 errors twice; cycle 2 is clean, so the budget must start over
    # instead of halting the run on the second faulty cycle.
    assert state.cycles == 4
    assert state.halted_reason == "max_cycles reached"
    assert orchestrator.consecutive_errors == 0
    assert state.history[0]["task_statuses"] == {"t1": "failed"}
    assert state.history[1]["task_statuses"] == {"t1": "completed"}


async def test_a_failed_task_is_retried_once_and_never_counts_as_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(
        tmp_path,
        monkeypatch,
        worker_script=[RuntimeError("worker exploded")],
        max_cycles=1,
        max_task_attempts=2,
    )

    state = await run_for(orchestrator)

    # The tests "passed", yet the run must not claim the goal is done.
    assert state.done is False
    assert state.halted_reason == "max_cycles reached"
    assert [task.status for task in state.tasks] == [TaskStatus.FAILED]
    assert state.tasks[0].attempts == 2
    assert "worker exploded" in (state.tasks[0].result or "")
    ends = [event for event in drain(queue) if event.kind == "task_end"]
    assert [event.data["attempt"] for event in ends] == [1, 2]
    assert {event.data["status"] for event in ends} == {"failed"}


async def test_a_task_is_never_retried_beyond_max_task_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        worker_script=[RuntimeError("worker exploded")],
        max_cycles=3,
        max_task_attempts=2,
        stagnation_limit=0,
    )

    state = await run_for(orchestrator)

    attempts = [event.data["attempt"] for event in drain(queue) if event.kind == "task_end"]
    assert attempts == [1, 2, 1, 2, 1, 2]
    assert state.tasks[0].attempts == 2


async def test_max_task_attempts_one_disables_the_retry_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(
        tmp_path,
        monkeypatch,
        worker_script=[RuntimeError("worker exploded")],
        max_cycles=1,
        max_task_attempts=1,
    )

    state = await run_for(orchestrator)

    assert state.tasks[0].attempts == 1
    assert [event.data["attempt"] for event in drain(queue) if event.kind == "task_end"] == [1]


def test_is_done_requires_tasks_and_a_passing_suite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, _ = build(tmp_path, monkeypatch, max_cycles=1)

    # The old ``_get_tasks()`` always returned [], which made ``_is_done()``
    # trivially true and ended the "infinite" loop after a single cycle.
    assert orchestrator.tasks == []
    assert orchestrator._is_done() is False  # noqa: SLF001

    orchestrator.tasks = [TaskModel(id="t1", description="x", status=TaskStatus.PENDING)]
    assert orchestrator._is_done() is False  # noqa: SLF001

    orchestrator.tasks[0].status = TaskStatus.COMPLETED
    orchestrator._test_passed = True  # noqa: SLF001
    assert orchestrator._is_done() is True  # noqa: SLF001

    orchestrator._test_passed = False  # noqa: SLF001
    assert orchestrator._is_done() is False  # noqa: SLF001

    orchestrator.tasks[0].status = TaskStatus.FAILED
    orchestrator._test_passed = True  # noqa: SLF001
    assert orchestrator._is_done() is False  # noqa: SLF001

    orchestrator.state.done = True
    assert orchestrator._is_done() is True  # noqa: SLF001


async def test_empty_plan_with_nothing_pending_finishes_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(tmp_path, monkeypatch, max_cycles=0)
    orchestrator.planner = _EmptyPlanner()

    state = await run_for(orchestrator)

    assert state.done is True
    assert state.halted_reason is None
    assert state.cycles == 1
    assert state.tasks == []
    messages = [event.message for event in drain(queue)]
    assert any("nothing is left to fix" in message for message in messages)


class _EmptyPlanner:
    """A planner whose plan holds no tasks at all."""

    async def plan(self, goal: str, context: str = "") -> Plan:
        return Plan(tasks=[], rationale="nothing left to do")


# ------------------------------------------------------------------ concurrency


async def test_max_parallel_tasks_caps_the_workers_in_flight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    five_tasks: dict[str, Any] = {
        "tasks": [{"id": f"t{index}", "description": f"task {index}"} for index in range(5)],
        "rationale": "five things",
    }
    orchestrator, _, _ = build(
        tmp_path,
        monkeypatch,
        plans=[five_tasks],
        max_cycles=1,
        max_parallel_tasks=2,
    )
    # The real planner trims a plan to ``max_parallel_tasks`` tasks, which would
    # hide the semaphore under test. Keep all five so the run's own cap is measured.
    monkeypatch.setattr(PlannerAgent, "_task_limit", lambda self, count: count)
    tracker = ConcurrencyTracker()
    monkeypatch.setattr(WorkerAgent, "run", tracker.worker)

    state = await run_for(orchestrator)

    assert len(state.tasks) == 5
    assert tracker.started == 5
    assert tracker.peak == 2
    assert tracker.peak <= 2
    assert state.done is True
    assert all(task.status is TaskStatus.COMPLETED for task in state.tasks)


class ConcurrencyTracker:
    """Records how many ``WorkerAgent.run`` coroutines overlap in time."""

    def __init__(self) -> None:
        self.current = 0
        self.peak = 0
        self.started = 0

    async def worker(self) -> str:
        self.current += 1
        self.started += 1
        self.peak = max(self.peak, self.current)
        try:
            await asyncio.sleep(0.01)
            return "worker finished the task"
        finally:
            self.current -= 1


# ----------------------------------------------------------------------- events


async def test_event_stream_is_ordered_and_json_serializable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(tmp_path, monkeypatch, max_cycles=1)

    state = await run_for(orchestrator)
    stream = drain(queue)
    kinds = [event.kind for event in stream]

    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_end"
    for required in ("cycle_start", "plan", "task_start", "task_end", "test_result", "cycle_end"):
        assert required in kinds, required
    assert kinds.index("run_start") < kinds.index("cycle_start") < kinds.index("plan")
    assert kinds.index("plan") < kinds.index("task_start") < kinds.index("task_end")
    assert kinds.index("task_end") < kinds.index("test_result") < kinds.index("cycle_end")
    assert kinds.index("cycle_end") < kinds.index("run_end")
    for event in stream:
        assert json.loads(json.dumps(event.as_dict())) == event.as_dict()
    assert stream[0].message == f"Goal: {GOAL}"
    assert stream[-1].data["done"] is True
    assert stream[-1].data["halted_reason"] is None
    assert stream[-1].data["cycles"] == state.cycles
    plan_event = stream[kinds.index("plan")]
    assert plan_event.data["tasks"][0]["id"] == "t1"


# ------------------------------------------------------------------ persistence


async def test_state_is_written_to_the_store_every_cycle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, _ = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=3,
        stagnation_limit=0,
    )
    store = orchestrator.store
    assert isinstance(store, RecordingStore)
    path = store.path

    state = await run_for(orchestrator)

    assert store.saves >= state.cycles
    assert path.is_file()
    persisted = StateStore(path).load()
    assert persisted is not None
    assert persisted.model_dump(mode="json") == state.model_dump(mode="json")
    assert persisted.goal == GOAL
    assert persisted.halted_reason == "max_cycles reached"
    assert len(persisted.history) == 3


async def test_a_read_only_workspace_does_not_kill_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(tmp_path, monkeypatch, max_cycles=1)
    store = orchestrator.store

    def broken_save(state: RunState) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(store, "save", broken_save)

    state = await run_for(orchestrator)

    assert state.done is True
    assert any(
        "could not save state" in event.message
        for event in drain(queue)
        if event.kind == "error"
    )


# ------------------------------------------------------------------- stop/abort


async def test_stop_from_a_subscriber_ends_the_run_promptly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, _ = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=0,  # unbounded: only stop() can end this run
        stagnation_limit=0,
        delay=0.001,
    )
    queue = orchestrator.events.subscribers[0]

    async def stop_after_first_cycle_start() -> None:
        while True:
            event = await queue.get()
            if event.kind == "cycle_start":
                await orchestrator.stop()
                return

    watcher = asyncio.create_task(stop_after_first_cycle_start())
    try:
        state = await asyncio.wait_for(orchestrator.run(), timeout=15)
    finally:
        watcher.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await watcher

    assert state.halted_reason == "stop requested by user"
    assert state.done is False
    assert state.cycles == 1
    assert orchestrator.is_running is False


async def test_stop_before_run_prevents_any_cycle_from_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, client, queue = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=0,  # unbounded: only stop() can end this run
        stagnation_limit=0,
    )

    await orchestrator.stop()
    state = await run_for(orchestrator)

    assert state.halted_reason == "stop requested by user"
    assert state.done is False
    assert state.cycles == 0
    assert orchestrator.cycles == 0
    assert orchestrator.is_running is False
    assert client.prompts == []  # no planner, worker or tester LLM call happened
    assert not any(event.kind == "cycle_start" for event in drain(queue))


async def test_cancelling_the_run_propagates_cancelled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(tmp_path, monkeypatch, max_cycles=0, delay=5.0)

    task = asyncio.create_task(orchestrator.run())
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(task), timeout=10)

    assert orchestrator.is_running is False
    assert orchestrator.state.halted_reason == "run cancelled"
    kinds = [event.kind for event in drain(queue)]
    assert kinds[0] == "run_start"
    assert kinds[-1] == "run_end"


async def test_an_unexpected_cycle_error_never_kills_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orchestrator, _, queue = build(
        tmp_path,
        monkeypatch,
        verdicts=FAIL,
        max_cycles=2,
        stagnation_limit=0,
    )

    async def exploding_cycle() -> None:
        raise ZeroDivisionError("division by zero")

    orchestrator._cycle = exploding_cycle  # type: ignore[method-assign]  # noqa: SLF001

    state = await run_for(orchestrator)

    assert state.cycles == 2
    assert state.halted_reason == "max_cycles reached"
    assert any(
        "ZeroDivisionError" in event.message for event in drain(queue) if event.kind == "error"
    )
    assert all("cycle aborted before tests ran" in entry["summary"] for entry in state.history)