"""Tests for furrow.core.orchestrator.

Ported from /tmp/orch_check.py with each scenario's assertions split into
focused, hermetic test functions.  No network or subprocess is involved:
planner, worker, and tester agents are replaced by injected fakes.
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import Any

import pytest
from rich.console import Console

from furrow.config import Plan, Settings, TaskModel, TaskStatus, TestResult
from furrow.core import orchestrator as orch
from furrow.core.orchestrator import (
    HALT_ERROR,
    HALT_GOAL_COMPLETE,
    HALT_MAX_CYCLES,
    HALT_NO_PROGRESS,
    HALT_NO_TASKS,
    STALL_LIMIT,
    Orchestrator,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

FAILURES = ["tests/test_x.py::test_y - AssertionError: expected 1 got 2"]

EXPECTED_KEYS: dict[str, set[str]] = {
    "cycle_start": {"type", "cycle", "goal"},
    "plan": {"type", "tasks", "rationale"},
    "task_start": {"type", "task_id"},
    "task_complete": {"type", "task_id", "summary"},
    "task_failed": {"type", "task_id", "error"},
    "test_result": {"type", "passed", "summary", "failures"},
    "goal_updated": {"type", "goal"},
    "error": {"type", "message"},
    "done": {"type", "reason", "cycles"},
}


class FakeClient:
    """Stands in for LLMClient; deliberately has no ``aclose``."""


class ClosableClient:
    def __init__(self) -> None:
        self.closed = 0

    async def aclose(self) -> None:
        self.closed += 1


class FakePlanner:
    def __init__(self, plans: list[Any]) -> None:
        self.plans = plans
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        item = self.plans[min(self.calls - 1, len(self.plans) - 1)]
        if isinstance(item, BaseException):
            raise item
        return item


class Recorder:
    def __init__(self) -> None:
        self.started: list[str] = []
        self.finished: list[str] = []
        self.live = 0
        self.peak = 0
        self.fail_ids: set[str] = set()
        self.cancel_ids: set[str] = set()
        self.goals_seen: list[str] = []


def _make_recorder() -> Recorder:
    return Recorder()


def _make_fakes(
    recorder: Recorder, tester_results: list[TestResult]
) -> tuple[type, type]:
    """Return (WorkerCls, TesterCls) bound to a specific recorder and result list."""

    class BoundWorker:
        def __init__(self, task: TaskModel, client: Any = None, settings: Any = None) -> None:
            self.task = task

        async def run(self) -> str:
            rec = recorder
            rec.started.append(self.task.id)
            rec.live += 1
            rec.peak = max(rec.peak, rec.live)
            try:
                await asyncio.sleep(0)
                if self.task.id in rec.cancel_ids:
                    raise asyncio.CancelledError()
                if self.task.id in rec.fail_ids:
                    raise RuntimeError(f"boom {self.task.id}")
                rec.finished.append(self.task.id)
                return f"did {self.task.id}"
            finally:
                rec.live -= 1

    class BoundTester:
        def __init__(self, client: Any = None, settings: Any = None) -> None:
            pass

        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            rec = recorder
            rec.goals_seen.append(goal)
            index = min(len(rec.goals_seen) - 1, len(tester_results) - 1)
            return tester_results[index]

    return BoundWorker, BoundTester


@pytest.fixture
def _patch_agents(monkeypatch: pytest.MonkeyPatch):
    """Monkeypatch WorkerAgent and TesterAgent; return the factory."""

    def _patch(rec: Recorder, results: list[TestResult]):
        WorkerCls, TesterCls = _make_fakes(rec, results)
        monkeypatch.setattr(orch, "WorkerAgent", WorkerCls)
        monkeypatch.setattr(orch, "TesterAgent", TesterCls)
        return WorkerCls, TesterCls

    return _patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def task(tid: str, deps: tuple[str, ...] = (), files: tuple[str, ...] = ("a.py",)) -> TaskModel:
    return TaskModel(
        id=tid, description=f"work on {tid}", files=list(files), dependencies=list(deps)
    )


def plan(*tasks: TaskModel, rationale: str = "because") -> Plan:
    return Plan(tasks=list(tasks), rationale=rationale)


def _build(
    plans: list[Any],
    *,
    max_cycles: int = 0,
    max_parallel_tasks: int = 5,
    goal: str = "ship the thing",
    on_event: Any = None,
    client: Any = None,
) -> tuple[Orchestrator, io.StringIO]:
    buffer = io.StringIO()
    orchestrator = Orchestrator(
        goal=goal,
        # ``FakeClient`` is a deliberate test double, not an ``LLMClient`` subclass.
        client=client if client is not None else FakeClient(),  # type: ignore[arg-type]
        # ``_env_file`` is a ``BaseSettings`` constructor keyword that works at runtime
        # but that mypy cannot see on the generated ``__init__`` signature.
        settings=Settings(  # type: ignore[call-arg]
            max_cycles=max_cycles,
            max_parallel_tasks=max_parallel_tasks,
            _env_file=None,
        ),
        console=Console(file=buffer, width=100),
        on_event=on_event,
    )
    orchestrator.planner = FakePlanner(plans)  # type: ignore[assignment]
    return orchestrator, buffer


def _done_reason(orchestrator: Orchestrator) -> str:
    return [e for e in orchestrator.events if e["type"] == "done"][-1]["reason"]


# ---------------------------------------------------------------------------
# Scenario a: success halts after one cycle
# ---------------------------------------------------------------------------


async def test_success_halts_one_cycle_goal_complete(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("t1"), task("t2"))], max_cycles=0)
    await orchestrator.run()
    assert orchestrator.cycles == 1
    assert _done_reason(orchestrator) == HALT_GOAL_COMPLETE


async def test_success_all_tasks_completed_and_is_done_true(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("t1"), task("t2"))], max_cycles=0)
    await orchestrator.run()
    assert all(t.status is TaskStatus.COMPLETED for t in orchestrator.tasks)
    assert orchestrator._is_done() is True


async def test_success_event_order_cycle_start_first_done_last(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("t1"), task("t2"))], max_cycles=0)
    await orchestrator.run()
    types = [e["type"] for e in orchestrator.events]
    assert types[0] == "cycle_start"
    assert types[-1] == "done"
    assert types.count("plan") == 1
    assert types.count("task_start") == 2
    assert types.count("task_complete") == 2


# ---------------------------------------------------------------------------
# Scenario b: failing tests keep looping
# ---------------------------------------------------------------------------


async def test_failing_tests_do_not_halt_after_one_cycle(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task(f"t{c}a"), task(f"t{c}b", deps=(f"t{c}a",))) for c in range(1, 4)],
        max_cycles=3,
    )
    await orchestrator.run()
    assert orchestrator.cycles == 3


async def test_failing_tests_halt_reason_is_max_cycles(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task(f"t{c}a"), task(f"t{c}b", deps=(f"t{c}a",))) for c in range(1, 4)],
        max_cycles=3,
    )
    await orchestrator.run()
    assert _done_reason(orchestrator) == HALT_MAX_CYCLES


async def test_failing_tests_is_done_false(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task(f"t{c}a"), task(f"t{c}b", deps=(f"t{c}a",))) for c in range(1, 4)],
        max_cycles=3,
    )
    await orchestrator.run()
    assert orchestrator._is_done() is False


async def test_failing_tests_goal_updated_for_every_cycle(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task(f"t{c}a"), task(f"t{c}b", deps=(f"t{c}a",))) for c in range(1, 4)],
        max_cycles=3,
    )
    await orchestrator.run()
    updates = [e for e in orchestrator.events if e["type"] == "goal_updated"]
    assert len(updates) == 3


async def test_failing_tests_original_goal_survives_verbatim(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task(f"t{c}a"), task(f"t{c}b", deps=(f"t{c}a",))) for c in range(1, 4)],
        max_cycles=3,
    )
    await orchestrator.run()
    updates = [e for e in orchestrator.events if e["type"] == "goal_updated"]
    assert all(
        e["goal"].startswith(orchestrator.original_goal) and f"- {FAILURES[0]}" in e["goal"]
        for e in updates
    )


async def test_failing_tests_self_goal_starts_with_original(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task(f"t{c}a"), task(f"t{c}b", deps=(f"t{c}a",))) for c in range(1, 4)],
        max_cycles=3,
    )
    await orchestrator.run()
    assert orchestrator.goal.startswith("ship the thing")
    assert orchestrator.original_goal == "ship the thing"


async def test_failing_tests_test_result_event_reports_failure(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task(f"t{c}a"), task(f"t{c}b", deps=(f"t{c}a",))) for c in range(1, 4)],
        max_cycles=3,
    )
    await orchestrator.run()
    assert all(
        e["passed"] is False and e["failures"] == FAILURES
        for e in orchestrator.events
        if e["type"] == "test_result"
    )


async def test_failing_tests_dependent_starts_after_dependency(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task(f"t{c}a"), task(f"t{c}b", deps=(f"t{c}a",))) for c in range(1, 4)],
        max_cycles=3,
    )
    await orchestrator.run()
    assert rec.started.index("t1b") > rec.finished.index("t1a")


# ---------------------------------------------------------------------------
# Scenario b2: empty failures keep the goal
# ---------------------------------------------------------------------------


async def test_empty_failures_goal_updated_twice(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="suite red", failures=[])])
    orchestrator, _ = _build([plan(task("t1"))], max_cycles=2)
    await orchestrator.run()
    updates = [e for e in orchestrator.events if e["type"] == "goal_updated"]
    assert len(updates) == 2


async def test_empty_failures_original_goal_preserved(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="suite red", failures=[])])
    orchestrator, _ = _build([plan(task("t1"))], max_cycles=2)
    await orchestrator.run()
    updates = [e for e in orchestrator.events if e["type"] == "goal_updated"]
    assert all(orchestrator.original_goal in e["goal"] for e in updates)


async def test_empty_failures_no_individual_failures_in_message(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="suite red", failures=[])])
    orchestrator, _ = _build([plan(task("t1"))], max_cycles=2)
    await orchestrator.run()
    updates = [e for e in orchestrator.events if e["type"] == "goal_updated"]
    assert "no individual failures" in updates[0]["goal"]


async def test_empty_failures_summary_in_message(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="suite red", failures=[])])
    orchestrator, _ = _build([plan(task("t1"))], max_cycles=2)
    await orchestrator.run()
    updates = [e for e in orchestrator.events if e["type"] == "goal_updated"]
    assert "suite red" in updates[0]["goal"]


# ---------------------------------------------------------------------------
# Scenario c: max_cycles is honoured
# ---------------------------------------------------------------------------


async def test_max_cycles_stops_at_exactly_two(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build([plan(task("t1"), task("t2"))], max_cycles=2)
    await orchestrator.run()
    assert orchestrator.cycles == 2


async def test_max_cycles_done_event_carries_reason_and_count(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build([plan(task("t1"), task("t2"))], max_cycles=2)
    await orchestrator.run()
    done_events = [e for e in orchestrator.events if e["type"] == "done"]
    assert _done_reason(orchestrator) == HALT_MAX_CYCLES
    assert done_events[-1]["cycles"] == 2


# ---------------------------------------------------------------------------
# Scenario d: empty plan halts
# ---------------------------------------------------------------------------


async def test_empty_plan_halts_with_no_tasks(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(rationale="nothing to do")], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert orchestrator.cycles == 1
    assert _done_reason(orchestrator) == HALT_NO_TASKS


async def test_empty_plan_tester_never_ran(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(rationale="nothing to do")], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert rec.goals_seen == []


# ---------------------------------------------------------------------------
# Scenario e: raising planner is clean
# ---------------------------------------------------------------------------


async def test_raising_planner_no_exception_escapes(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([RuntimeError("planner exploded")], max_cycles=0)
    raised = False
    try:
        await asyncio.wait_for(orchestrator.run(), timeout=5)
    except BaseException:
        raised = True
    assert not raised


async def test_raising_planner_emits_error_and_halts(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([RuntimeError("planner exploded")], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    errors = [e for e in orchestrator.events if e["type"] == "error"]
    assert len(errors) == 1
    assert "Planning failed" in errors[0]["message"]
    assert _done_reason(orchestrator) == HALT_ERROR


async def test_raising_planner_no_traceback_printed(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, buf = _build([RuntimeError("planner exploded")], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert "Traceback" not in buf.getvalue()


async def test_error_event_keys_match_contract(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([RuntimeError("planner exploded")], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    error_events = [e for e in orchestrator.events if e["type"] == "error"]
    assert len(error_events) == 1
    assert set(error_events[0].keys()) == {"type", "message"}


# ---------------------------------------------------------------------------
# Scenario f: events are json serialisable
# ---------------------------------------------------------------------------


async def test_events_are_non_empty(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="red", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task("t1", deps=("t2",)), task("t2"))], max_cycles=0
    )
    await orchestrator.run()
    assert len(orchestrator.events) > 0


async def test_events_are_json_dumps_able(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="red", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task("t1", deps=("t2",)), task("t2"))], max_cycles=0
    )
    await orchestrator.run()
    blobs = [json.dumps(event) for event in orchestrator.events]
    assert len(blobs) == len(orchestrator.events)


async def test_events_type_and_keys_match_contract(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="red", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task("t1", deps=("t2",)), task("t2"))], max_cycles=0
    )
    await orchestrator.run()
    bad = [
        e["type"]
        for e in orchestrator.events
        if e["type"] not in EXPECTED_KEYS or set(e) != EXPECTED_KEYS[e["type"]]
    ]
    assert not bad, str(bad)


async def test_plan_task_fields_match_contract(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="red", failures=FAILURES)])
    orchestrator, _ = _build(
        [plan(task("t1", deps=("t2",)), task("t2"))], max_cycles=0
    )
    await orchestrator.run()
    assert all(
        set(item) == {"id", "description", "files", "dependencies"}
        for e in orchestrator.events
        if e["type"] == "plan"
        for item in e["tasks"]
    )


# ---------------------------------------------------------------------------
# Scenario g: unlimited but stalled halts
# ---------------------------------------------------------------------------


async def test_stalled_halts_after_stall_limit_plus_one(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build([plan(task("t1"), task("t2"))], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=10)
    assert _done_reason(orchestrator) == HALT_NO_PROGRESS
    assert orchestrator.cycles == STALL_LIMIT + 1


async def test_stalled_finished_work_not_rerun(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=False, summary="1 failed", failures=FAILURES)])
    orchestrator, _ = _build([plan(task("t1"), task("t2"))], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=10)
    assert rec.started == ["t1", "t2"]


# ---------------------------------------------------------------------------
# Scenario h: dependency levels
# ---------------------------------------------------------------------------


async def test_dependency_levels_a_first(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build(
        [
            plan(
                task("b", deps=("a",)),
                task("a"),
                task("c", deps=("a",)),
                task("d", deps=("b", "c")),
            )
        ],
        max_cycles=0,
        max_parallel_tasks=4,
    )
    await orchestrator.run()
    assert rec.started[0] == "a"
    assert rec.started[-1] == "d"


async def test_dependency_levels_b_and_c_after_a(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build(
        [
            plan(
                task("b", deps=("a",)),
                task("a"),
                task("c", deps=("a",)),
                task("d", deps=("b", "c")),
            )
        ],
        max_cycles=0,
        max_parallel_tasks=4,
    )
    await orchestrator.run()
    assert rec.started.index("b") > rec.finished.index("a")
    assert rec.started.index("c") > rec.finished.index("a")
    assert rec.started.index("d") > max(rec.finished.index("b"), rec.finished.index("c"))


async def test_dependency_levels_siblings_concurrent(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build(
        [
            plan(
                task("b", deps=("a",)),
                task("a"),
                task("c", deps=("a",)),
                task("d", deps=("b", "c")),
            )
        ],
        max_cycles=0,
        max_parallel_tasks=4,
    )
    await orchestrator.run()
    assert rec.peak == 2


async def test_dependency_levels_completes_goal_complete(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build(
        [
            plan(
                task("b", deps=("a",)),
                task("a"),
                task("c", deps=("a",)),
                task("d", deps=("b", "c")),
            )
        ],
        max_cycles=0,
        max_parallel_tasks=4,
    )
    await orchestrator.run()
    assert _done_reason(orchestrator) == HALT_GOAL_COMPLETE


# ---------------------------------------------------------------------------
# Scenario i: dependency cycle does not hang
# ---------------------------------------------------------------------------


async def test_dependency_cycle_runs_and_finishes(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build(
        [plan(task("a", deps=("b",)), task("b", deps=("a",)))],
        max_cycles=0,
    )
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert _done_reason(orchestrator) == HALT_GOAL_COMPLETE
    assert sorted(rec.started) == ["a", "b"]


async def test_dependency_cycle_reported_in_output(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, buf = _build(
        [plan(task("a", deps=("b",)), task("b", deps=("a",)))],
        max_cycles=0,
    )
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert "Dependency cycle" in buf.getvalue()


# ---------------------------------------------------------------------------
# Scenario j: unknown dependency runs
# ---------------------------------------------------------------------------


async def test_unknown_dependency_task_still_runs(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("a", deps=("ghost",)))], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert rec.started == ["a"]


async def test_unknown_dependency_reported_in_output(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, buf = _build([plan(task("a", deps=("ghost",)))], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert "unknown task id" in buf.getvalue()


# ---------------------------------------------------------------------------
# Scenario k: parallelism cap
# ---------------------------------------------------------------------------


async def test_parallelism_cap_never_exceeds_in_flight(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build(
        [plan(*[task(f"t{i}") for i in range(1, 7)])],
        max_cycles=0,
        max_parallel_tasks=2,
    )
    await orchestrator.run()
    assert rec.peak == 2


async def test_parallelism_cap_all_six_run(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build(
        [plan(*[task(f"t{i}") for i in range(1, 7)])],
        max_cycles=0,
        max_parallel_tasks=2,
    )
    await orchestrator.run()
    assert len(rec.started) == 6
    assert _done_reason(orchestrator) == HALT_GOAL_COMPLETE


# ---------------------------------------------------------------------------
# Scenario l: raising event sink
# ---------------------------------------------------------------------------


async def test_raising_sink_does_not_end_run(_patch_agents: Any) -> None:
    seen: list[dict[str, Any]] = []

    def sink(event: dict[str, Any]) -> None:
        seen.append(event)
        raise RuntimeError("websocket closed")

    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("t1"))], max_cycles=0, on_event=sink)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert _done_reason(orchestrator) == HALT_GOAL_COMPLETE


async def test_raising_sink_saw_every_event(_patch_agents: Any) -> None:
    seen: list[dict[str, Any]] = []

    def sink(event: dict[str, Any]) -> None:
        seen.append(event)
        raise RuntimeError("websocket closed")

    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("t1"))], max_cycles=0, on_event=sink)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert len(seen) == len(orchestrator.events)


async def test_raising_sink_failure_reported(_patch_agents: Any) -> None:
    seen: list[dict[str, Any]] = []

    def sink(event: dict[str, Any]) -> None:
        seen.append(event)
        raise RuntimeError("websocket closed")

    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, buf = _build([plan(task("t1"))], max_cycles=0, on_event=sink)
    await asyncio.wait_for(orchestrator.run(), timeout=5)
    assert "on_event sink raised" in buf.getvalue()


# ---------------------------------------------------------------------------
# Scenario m: merge keeps finished work
# ---------------------------------------------------------------------------


async def test_merge_keeps_finished_no_rerun(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(
        rec,
        [
            TestResult(passed=False, summary="1 failed", failures=FAILURES),
            TestResult(passed=True, summary="all green"),
        ],
    )
    orchestrator, _ = _build(
        [plan(task("a"), task("b")), plan(task("a"), task("b"), task("c"))],
        max_cycles=5,
    )
    await orchestrator.run()
    assert rec.started == ["a", "b", "c"]


async def test_merge_completed_results_survive(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(
        rec,
        [
            TestResult(passed=False, summary="1 failed", failures=FAILURES),
            TestResult(passed=True, summary="all green"),
        ],
    )
    orchestrator, _ = _build(
        [plan(task("a"), task("b")), plan(task("a"), task("b"), task("c"))],
        max_cycles=5,
    )
    await orchestrator.run()
    assert orchestrator.tasks[0].result == "did a"
    assert orchestrator.tasks[1].result == "did b"


async def test_merge_halts_goal_complete_cycle_two(_patch_agents: Any) -> None:
    rec = _make_recorder()
    _patch_agents(
        rec,
        [
            TestResult(passed=False, summary="1 failed", failures=FAILURES),
            TestResult(passed=True, summary="all green"),
        ],
    )
    orchestrator, _ = _build(
        [plan(task("a"), task("b")), plan(task("a"), task("b"), task("c"))],
        max_cycles=5,
    )
    await orchestrator.run()
    assert orchestrator.cycles == 2
    assert _done_reason(orchestrator) == HALT_GOAL_COMPLETE


# ---------------------------------------------------------------------------
# Scenario n: failed task is retried
# ---------------------------------------------------------------------------


async def test_failed_task_is_retried_completed_not_rerun(
    _patch_agents: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _make_recorder()
    WorkerCls, TesterCls = _patch_agents(
        rec,
        [
            TestResult(passed=False, summary="1 failed", failures=FAILURES),
            TestResult(passed=True, summary="all green"),
        ],
    )

    failed_once: set[str] = set()

    # ``WorkerCls`` is a class object rather than a type alias, so mypy rejects it as
    # a base class twice over; the fake is a deliberate test double.
    class FlakyWorker(WorkerCls):  # type: ignore[valid-type,misc]
        async def run(self) -> str:
            if self.task.id == "a" and "a" not in failed_once:
                failed_once.add("a")
                raise RuntimeError("boom a")
            return await super().run()

    monkeypatch.setattr(orch, "WorkerAgent", FlakyWorker)
    orchestrator, _ = _build(
        [plan(task("a"), task("b")) for _ in range(2)],
        max_cycles=5,
    )
    await orchestrator.run()
    # "a" raises before super().run() is reached, so it is recorded only on its
    # retry; the order within one cycle's asyncio.gather is not deterministic.
    # The intent: a COMPLETED task ("b") is never re-run, a FAILED one ("a") is.
    assert sorted(rec.started) == ["a", "b"]
    assert rec.started.count("b") == 1


async def test_retried_task_succeeds_and_completes(
    _patch_agents: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = _make_recorder()
    WorkerCls, TesterCls = _patch_agents(
        rec,
        [
            TestResult(passed=False, summary="1 failed", failures=FAILURES),
            TestResult(passed=True, summary="all green"),
        ],
    )

    failed_once: set[str] = set()

    # ``WorkerCls`` is a class object rather than a type alias, so mypy rejects it as
    # a base class twice over; the fake is a deliberate test double.
    class FlakyWorker(WorkerCls):  # type: ignore[valid-type,misc]
        async def run(self) -> str:
            if self.task.id == "a" and "a" not in failed_once:
                failed_once.add("a")
                raise RuntimeError("boom a")
            return await super().run()

    monkeypatch.setattr(orch, "WorkerAgent", FlakyWorker)
    orchestrator, _ = _build(
        [plan(task("a"), task("b")) for _ in range(2)],
        max_cycles=5,
    )
    await orchestrator.run()
    assert _done_reason(orchestrator) == HALT_GOAL_COMPLETE
    assert orchestrator.tasks[0].status is TaskStatus.COMPLETED
    assert orchestrator.tasks[0].result == "did a"


# ---------------------------------------------------------------------------
# Scenario o: cancelled worker is a failure
# ---------------------------------------------------------------------------


async def test_cancelled_worker_is_failed_not_summary(_patch_agents: Any) -> None:
    rec = _make_recorder()
    rec.cancel_ids = {"boom"}
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("boom"), task("fine"))], max_cycles=1)
    await orchestrator.run()
    assert orchestrator.tasks[0].status is TaskStatus.FAILED
    assert "CancelledError" in (orchestrator.tasks[0].result or "")


async def test_cancelled_worker_emits_task_failed_event(_patch_agents: Any) -> None:
    rec = _make_recorder()
    rec.cancel_ids = {"boom"}
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("boom"), task("fine"))], max_cycles=1)
    await orchestrator.run()
    assert any(
        e["type"] == "task_failed" and e["task_id"] == "boom"
        for e in orchestrator.events
    )


async def test_cancelled_worker_bounded_by_max_cycles(_patch_agents: Any) -> None:
    rec = _make_recorder()
    rec.cancel_ids = {"boom"}
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("boom"), task("fine"))], max_cycles=1)
    await orchestrator.run()
    assert _done_reason(orchestrator) == HALT_MAX_CYCLES


async def test_task_failed_event_keys_match_contract(_patch_agents: Any) -> None:
    rec = _make_recorder()
    rec.cancel_ids = {"boom"}
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("boom"), task("fine"))], max_cycles=1)
    await orchestrator.run()
    failed_events = [e for e in orchestrator.events if e["type"] == "task_failed"]
    assert len(failed_events) == 1
    assert set(failed_events[0].keys()) == {"type", "task_id", "error"}
    assert failed_events[0]["task_id"] == "boom"


# ---------------------------------------------------------------------------
# Scenario p: failed dependency blocks and stalls
# ---------------------------------------------------------------------------


async def test_failed_dependency_dependent_never_starts(_patch_agents: Any) -> None:
    rec = _make_recorder()
    rec.fail_ids = {"a"}
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("a"), task("b", deps=("a",)))], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=10)
    assert rec.started == ["a"] * orchestrator.cycles


async def test_failed_dependency_stays_pending_dependency_failed(_patch_agents: Any) -> None:
    rec = _make_recorder()
    rec.fail_ids = {"a"}
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("a"), task("b", deps=("a",)))], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=10)
    assert orchestrator.tasks[1].status is TaskStatus.PENDING
    assert orchestrator.tasks[0].status is TaskStatus.FAILED


async def test_failed_dependency_halts_no_progress(_patch_agents: Any) -> None:
    rec = _make_recorder()
    rec.fail_ids = {"a"}
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, _ = _build([plan(task("a"), task("b", deps=("a",)))], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=10)
    assert _done_reason(orchestrator) == HALT_NO_PROGRESS


async def test_failed_dependency_skip_reported(_patch_agents: Any) -> None:
    rec = _make_recorder()
    rec.fail_ids = {"a"}
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    orchestrator, buf = _build([plan(task("a"), task("b", deps=("a",)))], max_cycles=0)
    await asyncio.wait_for(orchestrator.run(), timeout=10)
    assert "did not complete" in buf.getvalue()


# ---------------------------------------------------------------------------
# Scenario q: aclose and ctor surface
# ---------------------------------------------------------------------------


async def test_aclose_tolerates_client_without_aclose() -> None:
    bare = Orchestrator(goal="x", client=FakeClient())  # type: ignore[arg-type]
    try:
        await bare.aclose()
    except BaseException:
        pytest.fail("aclose raised on client without aclose")


async def test_aclose_forwards_to_client_with_aclose() -> None:
    closable = ClosableClient()
    orch_q = Orchestrator(goal="x", client=closable)  # type: ignore[arg-type]
    await orch_q.aclose()
    assert closable.closed == 1


async def test_orchestrator_positional_goal_and_client() -> None:
    positional = Orchestrator("x", FakeClient())  # type: ignore[arg-type]
    assert positional.original_goal == "x"
    assert positional.client is not None


async def test_fresh_orchestrator_empty_state() -> None:
    positional = Orchestrator("x", FakeClient())  # type: ignore[arg-type]
    assert positional._get_tasks() == []
    assert positional.test_result is None
    assert positional.cycles == 0
    assert positional._is_done() is False


async def test_run_and_aclose_are_coroutine_functions() -> None:
    assert asyncio.iscoroutinefunction(Orchestrator.run)
    assert asyncio.iscoroutinefunction(Orchestrator.aclose)


# ---------------------------------------------------------------------------
# Extra: worker text alone does not count as progress
# ---------------------------------------------------------------------------


async def test_worker_result_text_alone_does_not_count_as_progress(
    monkeypatch: pytest.MonkeyPatch, _patch_agents: Any
) -> None:
    """Rewritten worker prose must not look like progress.

    ``StateToken`` is ``(completed, failed, test outcome)``; the worker's own
    wording is deliberately excluded because an LLM rewrites it on every cycle
    even when nothing was achieved.  A green verdict would make ``_is_done()``
    true after one cycle and halt before the stall detector, so the tester is
    stubbed red here.  On top of that the token is compared across two runs that
    differ *only* in the worker's wording, which is what actually pins the
    invariant down instead of merely reaching the stall branch.
    """
    red = TestResult(passed=False, summary="1 failed", failures=FAILURES)

    async def _stalled_token(prose: str) -> tuple[str | None, Any]:
        rec = _make_recorder()
        WorkerCls, _ = _patch_agents(rec, [red])

        # Deliberate test double; see the ``FlakyWorker`` sites above.
        class ProseWorker(WorkerCls):  # type: ignore[valid-type,misc]
            async def run(self) -> str:
                await super().run()
                return prose

        monkeypatch.setattr(orch, "WorkerAgent", ProseWorker)
        orchestrator, _ = _build([plan(task("t1"))], max_cycles=0)
        await asyncio.wait_for(orchestrator.run(), timeout=10)
        assert _done_reason(orchestrator) == HALT_NO_PROGRESS
        assert orchestrator.cycles == STALL_LIMIT + 1
        return orchestrator.tasks[0].result, orchestrator._state_token()

    first_result, first_token = await _stalled_token("tried once")
    second_result, second_token = await _stalled_token("tried twice, and worded differently")

    # The two runs really did store different worker prose ...
    assert (first_result, second_result) == (
        "tried once",
        "tried twice, and worded differently",
    )
    # ... and still fingerprinted identically, so the prose is excluded.
    assert first_token == second_token == (1, 0, (False, tuple(FAILURES)))


# ---------------------------------------------------------------------------
# Extra: tester explosion becomes a failing test result
# ---------------------------------------------------------------------------


async def test_tester_explosion_becomes_a_failing_test_result(
    monkeypatch: pytest.MonkeyPatch, _patch_agents: Any
) -> None:
    rec = _make_recorder()
    WorkerCls, TesterCls = _patch_agents(rec, [TestResult(passed=True, summary="all green")])

    # Deliberate test double; see the ``FlakyWorker`` sites above.
    class ExplodingTester(TesterCls):  # type: ignore[valid-type,misc]
        async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
            raise RuntimeError("tester exploded")

    monkeypatch.setattr(orch, "TesterAgent", ExplodingTester)
    orchestrator, _ = _build([plan(task("t1"), task("t2"))], max_cycles=0)
    await orchestrator.run()
    assert orchestrator.test_result is not None
    assert orchestrator.test_result.passed is False
    assert "Tester failed" in orchestrator.test_result.summary


# ---------------------------------------------------------------------------
# Extra: duplicate task ids keep the first copy
# ---------------------------------------------------------------------------


async def test_duplicate_task_ids_in_one_plan_keep_the_first_copy(
    _patch_agents: Any,
) -> None:
    rec = _make_recorder()
    _patch_agents(rec, [TestResult(passed=True, summary="all green")])
    t = task("t1")
    orchestrator, _ = _build([plan(t, t)], max_cycles=0)
    await orchestrator.run()
    assert rec.started == ["t1"]
    assert rec.finished == ["t1"]
    assert len(orchestrator.tasks) == 1
    assert orchestrator.tasks[0].id == "t1"
    assert _done_reason(orchestrator) == HALT_GOAL_COMPLETE
