"""Tests for ``furrow.core.orchestrator.Orchestrator``.

The orchestrator owns the plan -> execute -> test loop, so this is the
highest-value test file in the project: a regression like "the infinite
loop halts after exactly one cycle" must fail here, fast and
deterministically.

Every test drives the real loop with a scripted fake LLM client -- no
network, no API keys, no real subprocesses (``TesterAgent._run_tests``
is neutered) -- and bounds ``run()`` with ``asyncio.wait_for`` so a hang
fails the test instead of the whole suite.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from furrow.config import Plan, Settings, TaskModel, TaskStatus
from furrow.core.orchestrator import Orchestrator

GOAL = "Ship the infinite loop"

#: Generous bound: a regression should fail the test, not hang the suite.
RUN_TIMEOUT = 10.0


class MessageLog:
    """Async progress sink that records every message it receives."""

    def __init__(self) -> None:
        self.messages: list[str] = []

    async def __call__(self, message: str) -> None:
        self.messages.append(message)


class FakeLLMClient:
    """Deterministic LLM double that dispatches on prompt shape.

    ``PlannerAgent``, ``WorkerAgent`` and ``TesterAgent`` all funnel
    through ``client.complete``. Their call sites are told apart by the
    line each agent appends to its prompt: worker prompts carry
    "Files to touch:", tester prompts carry "Exit code:", and anything
    else is the planner.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.plans: list[Plan] = []
        self.planner_error: BaseException | None = None
        self.worker_output = "implementation summary"
        self.worker_error: BaseException | None = None
        # The tester's process exit code (see install_fake_tester) is the
        # authoritative pass/fail signal; this payload only supplies the
        # human-readable summary and the failure list fed back to the
        # planner. Defaults to failing so loops stay alive across cycles.
        self.test_report: dict[str, Any] = {
            "passed": False,
            "summary": "1 failing test",
            "failures": ["test_default failed"],
        }
        self.planner_prompts: list[str] = []
        self.worker_calls = 0
        self._active_workers = 0
        self.peak_workers = 0

    async def complete(
        self, prompt: str, system: str = "", model: str | None = None
    ) -> str:
        if "Files to touch:" in prompt:
            return await self._worker_complete()
        if "Exit code:" in prompt:
            return json.dumps(self.test_report)
        return await self._planner_complete(prompt)

    def add_plan(self, *task_ids: str) -> None:
        """Queue one plan; calling it with no ids scripts an empty plan."""
        self.plans.append(
            Plan(
                tasks=[
                    TaskModel(id=task_id, description=f"do {task_id}", files=["mod.py"])
                    for task_id in task_ids
                ],
                rationale="scripted plan",
            )
        )

    async def _planner_complete(self, prompt: str) -> str:
        self.planner_prompts.append(prompt)
        if self.planner_error is not None:
            raise self.planner_error
        if not self.plans:
            raise AssertionError("FakeLLMClient: no scripted plan left")
        return self.plans.pop(0).model_dump_json()

    async def _worker_complete(self) -> str:
        self.worker_calls += 1
        self._active_workers += 1
        self.peak_workers = max(self.peak_workers, self._active_workers)
        try:
            # Yield once so sibling workers under the concurrency cap get a
            # chance to start: this is what makes peak_workers meaningful.
            await asyncio.sleep(0)
            if self.worker_error is not None:
                raise self.worker_error
            return self.worker_output
        finally:
            self._active_workers -= 1


def install_fake_tester(monkeypatch: pytest.MonkeyPatch, exit_code: int = 0) -> None:
    """Neuter ``TesterAgent._run_tests`` so no real test subprocess spawns.

    Imported locally: ``TestRun``/``TesterAgent`` live in this
    module's namespace otherwise, and pytest tries to collect any
    name starting with "Test" as a test class.
    """
    from furrow.agents.tester import TesterAgent, TestRun

    async def _fake_run_tests(self: TesterAgent) -> TestRun:
        return TestRun(output="fake output", returncode=exit_code, command="fake")

    monkeypatch.setattr(TesterAgent, "_run_tests", _fake_run_tests)


def patch_orchestrator_settings(
    monkeypatch: pytest.MonkeyPatch, **overrides: Any
) -> Settings:
    """Replace the module-level settings the orchestrator reads per cycle.

    ``max_parallel_tasks``, ``max_task_retries`` and ``max_cycles`` are
    read from the global ``settings`` imported into the orchestrator
    module, so capping concurrency or cycles from a test means patching
    that reference (the fake client's own settings are unrelated).
    """
    values: dict[str, Any] = {
        "max_parallel_tasks": 5,
        "max_task_retries": 1,
        "max_cycles": 0,
    }
    values.update(overrides)
    patched = Settings(**values)
    monkeypatch.setattr("furrow.core.orchestrator.settings", patched)
    return patched


def script_cycles(client: FakeLLMClient, count: int) -> None:
    """Give the planner a fresh task for each of ``count`` cycles."""
    for cycle in range(count):
        client.add_plan(f"task-{cycle}")


# ---------------------------------------------------------------------- tests


async def test_loop_runs_many_cycles_and_not_just_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION: the loop must not halt after a single cycle.

    The old implementation treated an empty task list as "everything
    settled" (``all([])`` is vacuously True), so a run stopped after
    exactly one cycle no matter what the planner returned. With a
    planner that keeps proposing real work and a red test suite, the
    loop must keep cycling until the cap says otherwise.
    """
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch, max_cycles=3)
    client = FakeLLMClient()
    script_cycles(client, 3)
    orch = Orchestrator(GOAL, client=client)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.cycles > 1, "regression: loop halted after a single cycle"
    assert result.cycles == 3
    assert result.stopped_reason == "max_cycles"
    assert len(client.planner_prompts) == 3
    assert len(result.reports) == 3


async def test_empty_task_list_is_not_settled() -> None:
    """The direct fix for the old ``0 >= 0`` bug: nothing ran != done.

    An empty task list must not count as "all settled", and a fresh
    orchestrator must not report a halt reason before any cycle ran.
    """
    orch = Orchestrator(GOAL, client=FakeLLMClient())
    assert orch.tasks == []
    assert orch._all_settled() is False
    assert orch._done_reason() is None


async def test_empty_plan_halts_with_goal_complete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty task list from the planner means the goal is complete."""
    install_fake_tester(monkeypatch)
    patch_orchestrator_settings(monkeypatch)
    client = FakeLLMClient()
    client.add_plan()  # no task ids -> empty plan
    orch = Orchestrator(GOAL, client=client)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.stopped_reason == "goal_complete"
    assert result.cycles == 1
    assert result.tasks == []


async def test_max_cycles_ctor_arg_halts_after_exactly_n_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The constructor argument wins over the (unlimited) settings value."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch)  # settings.max_cycles=0: unlimited
    client = FakeLLMClient()
    script_cycles(client, 4)  # one spare: proves the cap, not the script, stops it
    orch = Orchestrator(GOAL, client=client, max_cycles=2)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.stopped_reason == "max_cycles"
    assert result.cycles == 2
    assert len(client.planner_prompts) == 2


async def test_max_cycles_from_settings_halts_after_exactly_n_cycles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no constructor argument, the global settings cap applies."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch, max_cycles=2)
    client = FakeLLMClient()
    script_cycles(client, 4)
    orch = Orchestrator(GOAL, client=client)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.stopped_reason == "max_cycles"
    assert result.cycles == 2


async def test_zero_max_cycles_means_unlimited_until_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_cycles=0 disables the cap entirely; only stop() ends the run."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch, max_cycles=0)
    client = FakeLLMClient()
    script_cycles(client, 5)
    log = MessageLog()
    orch = Orchestrator(GOAL, client=client)

    async def on_progress(message: str) -> None:
        log.messages.append(message)
        if "Cycle 2" in message:
            await orch.stop()

    orch.on_progress = on_progress

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.stopped_reason == "stopped"
    assert result.cycles == 2


async def test_stop_from_progress_callback_returns_promptly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop() called mid-run halts the loop with stopped_reason == "stopped"."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch)  # unlimited: only stop() can end it
    client = FakeLLMClient()
    script_cycles(client, 10)
    log = MessageLog()
    orch = Orchestrator(GOAL, client=client)

    async def on_progress(message: str) -> None:
        log.messages.append(message)
        if "Cycle 3" in message:
            await orch.stop()

    orch.on_progress = on_progress

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.stopped_reason == "stopped"
    assert result.cycles == 3
    assert orch.stopped is True


async def test_stop_from_another_task_halts_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """stop() from a concurrent task interrupts the in-flight cycle."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch)
    client = FakeLLMClient()
    script_cycles(client, 10)
    orch = Orchestrator(GOAL, client=client)

    async def stop_later() -> None:
        await asyncio.sleep(0)  # let run() get into its first cycle
        await orch.stop()

    stopper = asyncio.create_task(stop_later())

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)
    await stopper

    assert result.stopped_reason == "stopped"
    assert result.cycles >= 1
    assert orch.stopped is True


async def test_worker_concurrency_never_exceeds_max_parallel_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observed peak worker concurrency must respect the global cap."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch, max_parallel_tasks=2, max_cycles=1)
    client = FakeLLMClient()
    client.add_plan("a", "b", "c", "d", "e")  # five tasks, cap of two
    orch = Orchestrator(GOAL, client=client)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert client.worker_calls == 5
    assert client.peak_workers <= 2, "concurrency cap was exceeded"
    assert client.peak_workers == 2, "workers never ran in parallel"
    assert [task.status for task in result.tasks] == [TaskStatus.COMPLETED] * 5


async def test_tasks_are_deduplicated_and_settled_tasks_are_not_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tasks merge by id across cycles and never regress to pending."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch, max_cycles=2)
    client = FakeLLMClient()
    client.add_plan("t1")  # cycle 1: brand new task
    # cycle 2: re-proposes t1 (still "pending" in the plan!) plus new task t2
    client.add_plan("t1", "t2")
    orch = Orchestrator(GOAL, client=client)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert [task.id for task in result.tasks] == ["t1", "t2"]
    assert [task.id for task in orch.tasks] == ["t1", "t2"]
    by_id = {task.id: task for task in result.tasks}
    assert by_id["t1"].status is TaskStatus.COMPLETED  # not dragged back to pending
    assert by_id["t2"].status is TaskStatus.COMPLETED


async def test_planner_failure_halts_with_error_and_is_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A planner blowup is recorded as stopped_reason == "error", not raised."""
    install_fake_tester(monkeypatch)
    patch_orchestrator_settings(monkeypatch)
    client = FakeLLMClient()
    client.planner_error = RuntimeError("planner exploded")
    log = MessageLog()
    orch = Orchestrator(GOAL, client=client, on_progress=log)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.stopped_reason == "error"
    assert result.cycles == 1
    assert any("planner exploded" in message for message in log.messages)
    assert any("Last error" in message for message in log.messages)


def test_keyboard_interrupt_is_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KeyboardInterrupt must propagate out of run(), not be swallowed.

    asyncio re-raises KeyboardInterrupt out of the event loop
    driver (Task.__step) rather than through the await chain, so
    the loop is driven with asyncio.run() inside pytest.raises:
    the interrupt escapes the inner loop and is caught here.
    """
    install_fake_tester(monkeypatch)
    patch_orchestrator_settings(monkeypatch)
    client = FakeLLMClient()
    client.planner_error = KeyboardInterrupt()
    orch = Orchestrator(GOAL, client=client)

    with pytest.raises(KeyboardInterrupt):
        asyncio.run(orch.run())


async def test_progress_callback_receives_human_readable_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_progress sees the goal, the cycle banners and the halt summary."""
    install_fake_tester(monkeypatch)
    patch_orchestrator_settings(monkeypatch)
    client = FakeLLMClient()
    client.add_plan()  # empty plan: one cycle, then goal complete
    log = MessageLog()
    orch = Orchestrator(GOAL, client=client, on_progress=log)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.stopped_reason == "goal_complete"
    assert log.messages[0] == f"Goal: {GOAL}"
    assert "═══ Cycle 1 ═══" in log.messages
    assert any("Goal complete" in message for message in log.messages)


async def test_raising_progress_callback_never_crashes_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken sink (say, a closed websocket) must not take the loop down."""
    install_fake_tester(monkeypatch)
    patch_orchestrator_settings(monkeypatch)
    client = FakeLLMClient()
    client.add_plan()

    async def exploding(message: str) -> None:
        raise RuntimeError("websocket closed")

    orch = Orchestrator(GOAL, client=client, on_progress=exploding)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.stopped_reason == "goal_complete"
    assert result.cycles == 1


async def test_failing_task_is_retried_up_to_max_task_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising worker is retried max_task_retries times, then marked failed."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch, max_task_retries=2, max_cycles=1)
    client = FakeLLMClient()
    client.add_plan("flaky")
    client.worker_error = RuntimeError("worker exploded")
    log = MessageLog()
    orch = Orchestrator(GOAL, client=client, on_progress=log)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert client.worker_calls == 3  # one attempt plus two retries
    assert result.tasks[0].status is TaskStatus.FAILED
    assert result.tasks[0].result == "worker exploded"
    assert any("retrying (1/2)" in message for message in log.messages)


async def test_goal_is_preserved_when_tests_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RunResult.goal stays the ORIGINAL goal, never a "fix tests" goal."""
    install_fake_tester(monkeypatch, exit_code=1)
    patch_orchestrator_settings(monkeypatch, max_cycles=2)
    client = FakeLLMClient()
    client.test_report = {
        "passed": False,
        "summary": "2 failing tests",
        "failures": ["test_alpha failed", "test_beta failed"],
    }
    script_cycles(client, 2)
    orch = Orchestrator(GOAL, client=client)

    result = await asyncio.wait_for(orch.run(), timeout=RUN_TIMEOUT)

    assert result.goal == GOAL
    assert orch.goal == GOAL
    # Cycle 1 sees the bare goal; cycle 2 sees the goal plus the failure
    # context -- the goal itself is never replaced.
    assert client.planner_prompts[0].endswith(f"Goal: {GOAL}\n")
    assert GOAL in client.planner_prompts[1]
    assert "test_alpha failed" in client.planner_prompts[1]
    assert "Fix failing tests" not in client.planner_prompts[1]
