from __future__ import annotations

from unittest.mock import MagicMock

from furrow.config import Plan, Settings, TaskModel
from furrow.config import TestResult as TestResultModel
from furrow.core.orchestrator import Orchestrator


def _make_orch(goal: str = "test goal") -> Orchestrator:
    """Construct an Orchestrator with a stub LLM client (no real API calls)."""
    return Orchestrator(goal=goal, client=MagicMock())


# ---------------------------------------------------------------------------
# _is_done tests
# ---------------------------------------------------------------------------


def test_is_done_no_tasks() -> None:
    orch = _make_orch()
    assert orch.tasks == []
    assert orch._is_done() is True


def test_is_done_all_completed() -> None:
    orch = _make_orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_with_failed() -> None:
    orch = _make_orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_with_pending() -> None:
    orch = _make_orch()
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


# ---------------------------------------------------------------------------
# _record_task / task tracking across cycles
# ---------------------------------------------------------------------------


def test_record_task_dedupes_by_id() -> None:
    orch = _make_orch()

    t1 = TaskModel(id="1", description="first", status="completed")
    orch._record_task(t1)
    assert len(orch.tasks) == 1

    # Same id, different status — should replace, not append.
    t1_updated = TaskModel(id="1", description="first", status="failed")
    orch._record_task(t1_updated)
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "failed"


def test_record_task_appends_new_ids() -> None:
    orch = _make_orch()
    orch._record_task(TaskModel(id="1", description="a", status="completed"))
    orch._record_task(TaskModel(id="2", description="b", status="completed"))
    orch._record_task(TaskModel(id="3", description="c", status="completed"))
    assert [t.id for t in orch.tasks] == ["1", "2", "3"]


# ---------------------------------------------------------------------------
# max_cycles enforcement
# ---------------------------------------------------------------------------


class _NoopOrchestrator(Orchestrator):
    """Orchestrator that never finishes, used to test max_cycles."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cycle_calls = 0

    async def _cycle(self) -> None:
        self.cycle_calls += 1

    def _is_done(self) -> bool:
        return False


async def test_max_cycles_enforcement() -> None:
    orch = _NoopOrchestrator(goal="test", client=MagicMock())
    orch.settings = Settings(max_cycles=2)
    await orch.run()
    assert orch.cycles == 2
    assert orch.cycle_calls == 2


async def test_max_cycles_zero_runs_until_done() -> None:
    """max_cycles=0 means unlimited (no enforcement)."""

    class _DoneAfter3(Orchestrator):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = 0

        async def _cycle(self) -> None:
            self.calls += 1

        def _is_done(self) -> bool:
            return self.calls >= 3

    orch = _DoneAfter3(goal="test", client=MagicMock())
    orch.settings = Settings(max_cycles=0)
    await orch.run()
    assert orch.cycles == 3


# ---------------------------------------------------------------------------
# _cycle integration with stubbed planner/worker/tester
# ---------------------------------------------------------------------------


class _StubOrchestrator(Orchestrator):
    """Orchestrator with injectable planner/worker/tester behaviour."""

    def __init__(
        self,
        *args,
        plan: Plan | None = None,
        worker_result: str = "ok",
        test_result: TestResultModel | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._stub_plan = plan
        self._stub_worker_result = worker_result
        self._stub_test_result = test_result or TestResultModel(passed=True, summary="ok")
        self.planner = MagicMock()
        self.planner.plan = MagicMock(return_value=self._stub_plan)

    async def _cycle(self) -> None:
        # Minimal controlled cycle — exercises _record_task and _is_done logic.
        if self._stub_plan is None:
            return
        for task in self._stub_plan.tasks:
            task.status = "completed"
            task.result = self._stub_worker_result
            self._record_task(task)


async def test_cycle_records_tasks_and_completes() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b"),
        ],
        rationale="stub",
    )
    orch = _StubOrchestrator(goal="test", client=MagicMock(), plan=plan)
    await orch._cycle()
    assert len(orch.tasks) == 2
    assert all(t.status == "completed" for t in orch.tasks)
    assert orch._is_done() is True


async def test_cycle_with_failed_task_not_done() -> None:
    plan = Plan(
        tasks=[
            TaskModel(id="1", description="a", status="failed"),
        ],
        rationale="stub",
    )
    orch = _StubOrchestrator(goal="test", client=MagicMock(), plan=plan)
    # Simulate failure path manually.
    for task in plan.tasks:
        task.status = "failed"
        task.result = "boom"
        orch._record_task(task)
    assert orch._is_done() is False