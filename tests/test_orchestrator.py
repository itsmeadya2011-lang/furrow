"""Tests for furrow.core.orchestrator — cycle logic, task tracking, and dependency ordering."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


def _make_orchestrator(goal: str = "Test goal", **kwargs) -> Orchestrator:
    """Create an Orchestrator with a mock LLMClient to avoid real API calls."""
    client = LLMClient()
    return Orchestrator(goal=goal, client=client, **kwargs)


def _make_task(id: str, desc: str = "desc", deps: list[str] | None = None) -> TaskModel:
    return TaskModel(id=id, description=desc, dependencies=deps or [])


# ---------------------------------------------------------------------------
#  Topological tier ordering
# ---------------------------------------------------------------------------

def test_topological_tiers_no_dependencies():
    tasks = [_make_task("1"), _make_task("2"), _make_task("3")]
    tiers = Orchestrator._topological_tiers(tasks)
    assert len(tiers) == 1
    assert len(tiers[0]) == 3


def test_topological_tiers_with_dependencies():
    tasks = [
        _make_task("1"),
        _make_task("2", deps=["1"]),
        _make_task("3", deps=["1", "2"]),
    ]
    tiers = Orchestrator._topological_tiers(tasks)
    assert len(tiers) == 3
    assert tiers[0][0].id == "1"
    assert tiers[1][0].id == "2"
    assert tiers[2][0].id == "3"


def test_topological_tiers_parallel_within_tier():
    """Tasks with no dependencies between them should be in the same tier."""
    tasks = [
        _make_task("1"),
        _make_task("2"),
        _make_task("3", deps=["1", "2"]),
    ]
    tiers = Orchestrator._topological_tiers(tasks)
    assert len(tiers) == 2
    assert len(tiers[0]) == 2
    assert len(tiers[1]) == 1


def test_topological_tiers_circular_dependency():
    """Circular dependencies should be resolved by including all remaining tasks."""
    tasks = [
        _make_task("1", deps=["2"]),
        _make_task("2", deps=["1"]),
    ]
    tiers = Orchestrator._topological_tiers(tasks)
    # Should produce at least one tier containing both
    assert sum(len(t) for t in tiers) == 2


def test_topological_tiers_empty():
    assert Orchestrator._topological_tiers([]) == []


# ---------------------------------------------------------------------------
#  _is_done logic
# ---------------------------------------------------------------------------

def test_is_done_no_tasks_planned():
    orch = _make_orchestrator()
    orch._cycle_total_count = 0
    assert orch._is_done() is True


def test_is_done_all_tasks_passed_tests():
    orch = _make_orchestrator()
    orch._cycle_total_count = 3
    orch._cycle_failed_count = 0
    orch._test_passed = True
    assert orch._is_done() is True


def test_is_done_tasks_failed():
    orch = _make_orchestrator()
    orch._cycle_total_count = 3
    orch._cycle_failed_count = 1
    orch._test_passed = False
    assert orch._is_done() is False


def test_is_done_tasks_passed_but_tests_failed():
    orch = _make_orchestrator()
    orch._cycle_total_count = 3
    orch._cycle_failed_count = 0
    orch._test_passed = False
    assert orch._is_done() is False


# ---------------------------------------------------------------------------
#  Task history tracking
# ---------------------------------------------------------------------------

def test_get_tasks_returns_history():
    orch = _make_orchestrator()
    orch._task_history = [_make_task("1"), _make_task("2")]
    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"


def test_get_tasks_empty_initially():
    orch = _make_orchestrator()
    assert orch._get_tasks() == []


# ---------------------------------------------------------------------------
#  Integration: full cycle with mocked LLM
# ---------------------------------------------------------------------------

async def test_orchestrator_run_completes_when_no_tasks():
    """When planner returns no tasks, the loop should terminate after 1 cycle."""
    orch = _make_orchestrator()
    empty_plan = Plan(tasks=[], rationale="Goal already met")
    with (
        patch.object(orch.planner, "plan", new=AsyncMock(return_value=empty_plan)),
        patch.object(orch.tester, "run", new=AsyncMock(return_value=TestResult(passed=True, summary="ok"))),
    ):
        await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True


async def test_orchestrator_run_respects_max_cycles():
    """Orchestrator should stop when max_cycles is reached."""
    settings = Settings(max_cycles=2)
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="test", client=client)
    plan = Plan(tasks=[_make_task("1")], rationale="test")

    call_count = 0

    async def mock_plan(goal: str) -> Plan:
        return plan

    async def mock_worker_run() -> str:
        nonlocal call_count
        call_count += 1
        return f"result {call_count}"

    with (
        patch.object(orch.planner, "plan", new=mock_plan),
        patch.object(
            orch.tester, "run",
            new=AsyncMock(return_value=TestResult(passed=False, summary="fail", failures=["fail"])),
        ),
        patch("furrow.core.orchestrator.WorkerAgent") as MockWorker,
    ):
        MockWorker.return_value.run = mock_worker_run
        await orch.run()

    assert orch.cycles == 2
    assert orch.max_cycles == 2


async def test_orchestrator_run_tracks_task_history():
    """Tasks from each cycle should be appended to task history."""
    orch = _make_orchestrator()
    plan = Plan(tasks=[_make_task("1"), _make_task("2")], rationale="test")

    with (
        patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)),
        patch.object(
            orch.tester, "run",
            new=AsyncMock(return_value=TestResult(passed=True, summary="ok")),
        ),
        patch("furrow.core.orchestrator.WorkerAgent") as MockWorker,
    ):
        mock_instance = MockWorker.return_value
        mock_instance.run = AsyncMock(return_value="done")
        await orch.run()

    assert len(orch._get_tasks()) == 2
    assert orch._get_tasks()[0].status == "completed"
    assert orch._get_tasks()[1].status == "completed"


async def test_orchestrator_run_continues_on_test_failure():
    """If tests fail, the orchestrator should retry in the next cycle."""
    orch = _make_orchestrator()
    # First cycle: tests fail. Second cycle: no tasks → done.
    plans = [
        Plan(tasks=[_make_task("1")], rationale="first"),
        Plan(tasks=[], rationale="done"),
    ]
    plan_iter = iter(plans)

    async def mock_plan(goal: str) -> Plan:
        return next(plan_iter)

    results_by_cycle: list[TestResult] = [
        TestResult(passed=False, summary="fail", failures=["broken"]),
        TestResult(passed=True, summary="ok"),
    ]
    test_iter = iter(results_by_cycle)

    async def mock_test(*args, **kwargs) -> TestResult:
        return next(test_iter)

    with (
        patch.object(orch.planner, "plan", new=mock_plan),
        patch.object(orch.tester, "run", new=mock_test),
        patch("furrow.core.orchestrator.WorkerAgent") as MockWorker,
    ):
        MockWorker.return_value.run = AsyncMock(return_value="done")
        await orch.run()

    assert orch.cycles == 2
    assert "broken" in orch.goal


async def test_orchestrator_emits_output():
    """The on_output callback should receive progress messages."""
    received: list[str] = []

    def on_output(msg: str) -> None:
        received.append(msg)

    orch = Orchestrator(goal="test", on_output=on_output)
    plan = Plan(tasks=[], rationale="done")

    with (
        patch.object(orch.planner, "plan", new=AsyncMock(return_value=plan)),
        patch.object(
            orch.tester, "run",
            new=AsyncMock(return_value=TestResult(passed=True, summary="ok")),
        ),
    ):
        await orch.run()

    assert len(received) > 0
    assert any("Furrow" in msg for msg in received)
    assert any("Cycle 1" in msg for msg in received)
