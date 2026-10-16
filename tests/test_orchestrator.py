import asyncio
from pathlib import Path

import pytest

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


class FakeClient:
    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        return ""

    async def read_file(self, path: str | Path) -> str:
        return ""

    async def write_file(self, path: str | Path, content: str) -> None:
        return None

    def list_files(self, directory: str | Path) -> list[str]:
        return []


def _task(task_id: str, status: str = "pending") -> TaskModel:
    return TaskModel(id=task_id, description=f"task {task_id}", status=status)


def test_is_done_empty_tasks_is_done():
    orch = Orchestrator(goal="x", client=FakeClient())
    assert orch._is_done() is True


def test_is_done_all_completed_is_done():
    orch = Orchestrator(goal="x", client=FakeClient())
    orch._tasks = {"1": _task("1", "completed"), "2": _task("2", "completed")}
    assert orch._is_done() is True


def test_is_done_any_failed_not_done():
    orch = Orchestrator(goal="x", client=FakeClient())
    orch._tasks = {"1": _task("1", "completed"), "2": _task("2", "failed")}
    assert orch._is_done() is False


def test_is_done_any_pending_not_done():
    orch = Orchestrator(goal="x", client=FakeClient())
    orch._tasks = {"1": _task("1", "completed"), "2": _task("2", "pending")}
    assert orch._is_done() is False


def test_get_tasks_returns_tracked_tasks():
    orch = Orchestrator(goal="x", client=FakeClient())
    t1 = _task("1")
    orch._tasks = {"1": t1}
    assert orch._get_tasks() == [t1]


@pytest.mark.asyncio
async def test_cycle_registers_new_tasks_only_once():
    client = FakeClient()
    orch = Orchestrator(goal="x", client=client, settings=Settings(max_cycles=1))
    plan_task = TaskModel(id="1", description="add feature", status="pending")

    async def fake_plan(goal: str):
        from furrow.config import Plan
        return Plan(tasks=[plan_task], rationale="r")

    orch.planner.plan = fake_plan  # type: ignore[method-assign]
    await orch._cycle()
    assert len(orch._tasks) == 1
    # Running again with the same plan task id should not duplicate.
    await orch._cycle()
    assert len(orch._tasks) == 1
