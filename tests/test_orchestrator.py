"""Unit tests for the dependency-aware scheduling and lifecycle logic.

These tests exercise the orchestrator *without* an LLM by overriding the
``_run_task`` hook and the planner/tester path. They verify the critical fixes:
``_get_tasks``/``_is_done`` reflect real task state, ``max_cycles`` is enforced,
dependencies are respected, and ``max_parallel_tasks`` bounds concurrency.
"""

from __future__ import annotations

import asyncio

import pytest

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def _task(
    task_id: str, deps: list[str] | None = None, status: str = "pending"
) -> TaskModel:
    return TaskModel(
        id=task_id, description=f"task {task_id}", dependencies=deps or [], status=status
    )


def _stub_orchestrator(settings: Settings | None = None) -> Orchestrator:
    """Build an Orchestrator. LLMClient construction does not make network calls,
    so a real client is fine here; ``_run_task`` is overridden per-test."""
    return Orchestrator(goal="test goal", settings=settings)


# --------------------------------------------------------------------------- #
# _get_tasks / _is_done
# --------------------------------------------------------------------------- #
def test_get_tasks_returns_instance_state():
    orch = _stub_orchestrator()
    orch.all_tasks = [_task("1"), _task("2")]
    assert orch._get_tasks() is orch.all_tasks
    assert {t.id for t in orch._get_tasks()} == {"1", "2"}


def test_is_done_false_when_tasks_empty_is_old_bug_fixed():
    """The old ``_get_tasks`` returned [] so _is_done was True after one cycle.
    With real state, an empty cycle is still considered done (nothing to do)."""
    orch = _stub_orchestrator()
    orch.all_tasks = []
    assert orch._is_done() is True


def test_is_done_all_completed():
    orch = _stub_orchestrator()
    orch.all_tasks = [TaskModel(id="1", description="a", status="completed")]
    assert orch._is_done() is True


def test_is_done_not_done_when_any_failed():
    orch = _stub_orchestrator()
    orch.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_not_done_when_any_pending():
    orch = _stub_orchestrator()
    orch.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


# --------------------------------------------------------------------------- #
# _compute_waves (pure, testable without an LLM)
# --------------------------------------------------------------------------- #
def test_compute_waves_empty():
    orch = _stub_orchestrator()
    assert orch._compute_waves([]) == []


def test_compute_waves_no_dependencies_single_wave():
    orch = _stub_orchestrator()
    tasks = [_task("1"), _task("2"), _task("3")]
    waves = orch._compute_waves(tasks)
    assert len(waves) == 1
    assert [t.id for t in waves[0]] == ["1", "2", "3"]


def test_compute_waves_respects_dependencies():
    orch = _stub_orchestrator()
    tasks = [
        _task("1"),
        _task("2", deps=["1"]),
        _task("3", deps=["1"]),
        _task("4", deps=["2", "3"]),
    ]
    waves = orch._compute_waves(tasks)
    assert [t.id for t in waves[0]] == ["1"]
    assert sorted(t.id for t in waves[1]) == ["2", "3"]
    assert [t.id for t in waves[2]] == ["4"]


def test_compute_waves_rejects_unknown_dependency():
    orch = _stub_orchestrator()
    tasks = [_task("1", deps=["missing"])]
    with pytest.raises(ValueError, match="unknown task"):
        orch._compute_waves(tasks)


def test_compute_waves_rejects_cycle():
    orch = _stub_orchestrator()
    tasks = [_task("1", deps=["2"]), _task("2", deps=["1"])]
    with pytest.raises(ValueError, match="Unresolvable"):
        orch._compute_waves(tasks)


# --------------------------------------------------------------------------- #
# _execute_tasks (no LLM: override _run_task)
# --------------------------------------------------------------------------- #
class _NoLLMOrchestrator(Orchestrator):
    """Orchestrator that records execution order instead of calling an LLM."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(goal="test goal", settings=settings)
        self.order: list[str] = []
        self.fail: set[str] = set()

    async def _run_task(self, task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            await asyncio.sleep(0)
            self.order.append(task.id)
            if task.id in self.fail:
                raise RuntimeError(f"boom {task.id}")
            return f"result-{task.id}"


async def test_execute_tasks_runs_in_dependency_order():
    settings = Settings(max_parallel_tasks=4)
    orch = _NoLLMOrchestrator(settings=settings)
    tasks = [
        _task("a"),
        _task("b", deps=["a"]),
        _task("c", deps=["a"]),
        _task("d", deps=["b", "c"]),
    ]
    await orch._execute_tasks(tasks)
    assert orch.order[0] == "a"
    assert orch.order[-1] == "d"
    assert orch.order.index("b") < orch.order.index("d")
    assert orch.order.index("c") < orch.order.index("d")
    for t in tasks:
        assert t.status == "completed"
        assert t.result == f"result-{t.id}"


async def test_execute_tasks_marks_dependents_blocked_on_failure():
    settings = Settings(max_parallel_tasks=4)
    orch = _NoLLMOrchestrator(settings=settings)
    orch.fail = {"a"}
    tasks = [_task("a"), _task("b", deps=["a"]), _task("c", deps=["b"])]
    await orch._execute_tasks(tasks)
    a, b, c = {t.id: t for t in tasks}.values()
    assert a.status == "failed"
    assert b.status == "failed" and b.result == "Skipped: dependency failed"
    assert c.status == "failed" and c.result == "Skipped: dependency failed"
    assert "a" in orch.order  # a ran
    assert "b" not in orch.order and "c" not in orch.order  # dependents skipped


async def test_execute_tasks_enforces_max_parallel_tasks():
    settings = Settings(max_parallel_tasks=2)
    orch = _NoLLMOrchestrator(settings=settings)

    peak = {"value": 0}
    current = {"value": 0}
    base_run = orch._run_task

    async def tracking_run(task: TaskModel, semaphore: asyncio.Semaphore) -> str:
        async with semaphore:
            current["value"] += 1
            peak["value"] = max(peak["value"], current["value"])
            await asyncio.sleep(0.05)
            current["value"] -= 1
            return f"result-{task.id}"

    orch._run_task = tracking_run  # type: ignore[method-assign]
    tasks = [_task(str(i)) for i in range(6)]  # all independent -> one wave
    await orch._execute_tasks(tasks)
    assert peak["value"] == 2, f"expected peak concurrency 2, got {peak['value']}"
    assert base_run is not orch._run_task  # sanity


async def test_execute_tasks_propagates_exception_result():
    settings = Settings(max_parallel_tasks=3)
    orch = _NoLLMOrchestrator(settings=settings)
    orch.fail = {"2"}
    tasks = [_task("1"), _task("2"), _task("3", deps=["2"])]
    await orch._execute_tasks(tasks)
    by_id = {t.id: t for t in tasks}
    assert by_id["2"].status == "failed"
    assert by_id["3"].status == "failed"
    assert "boom 2" in by_id["2"].result
    assert by_id["1"].status == "completed"


# --------------------------------------------------------------------------- #
# max_cycles enforcement in run() (override _cycle to avoid the LLM)
# --------------------------------------------------------------------------- #
class _StubCycleOrchestrator(Orchestrator):
    """Records cycle calls and never reports itself as done."""

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(goal="test goal", settings=settings)
        self.cycles_run = 0

    async def _cycle(self) -> None:
        self.cycles_run += 1
        self.all_tasks = [_task("1", status="pending")]

    def _is_done(self) -> bool:
        return False


async def test_run_respects_max_cycles():
    settings = Settings(max_cycles=3)
    orch = _StubCycleOrchestrator(settings=settings)
    await orch.run()
    assert orch.cycles_run == 3
    assert orch.cycles == 3


async def test_run_unlimited_without_max_cycles():
    # Without a cycle cap, run() would loop forever; we cap via a failing
    # _is_done after a few cycles using an override.
    settings = Settings(max_cycles=0)
    orch = _StubCycleOrchestrator(settings=settings)

    real_is_done = orch._is_done

    def once_done() -> bool:
        if orch.cycles_run >= 5:
            return True
        return real_is_done()

    orch._is_done = once_done  # type: ignore[method-assign]
    await orch.run()
    assert orch.cycles_run == 5
