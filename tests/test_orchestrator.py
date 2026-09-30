from __future__ import annotations

import asyncio
from unittest import mock

from furrow.config import Plan, TaskModel, TestResult, settings
from furrow.core import orchestrator as orch_module
from furrow.core.orchestrator import Orchestrator


def _tasks(statuses: list[str]) -> list[TaskModel]:
    tasks: list[TaskModel] = []
    for i, s in enumerate(statuses):
        t = TaskModel(id=str(i), description=f"task {i}")
        t.status = s
        tasks.append(t)
    return tasks


def _plan(statuses: list[str]) -> Plan:
    return Plan(tasks=_tasks(statuses), rationale="ok")


def _make_client() -> mock.MagicMock:
    client = mock.MagicMock()
    client.settings = mock.MagicMock()
    client.complete = mock.AsyncMock(return_value="{}")
    return client


def _patch_agents(plan: Plan | None = None, test_result: TestResult | None = None) -> tuple:
    planner_mock = mock.MagicMock()
    worker_mock = mock.MagicMock()
    tester_mock = mock.MagicMock()
    if plan is not None:
        planner_mock.return_value.plan = mock.AsyncMock(return_value=plan)
    if test_result is not None:
        tester_mock.return_value.run = mock.AsyncMock(return_value=test_result)
    return planner_mock, worker_mock, tester_mock


async def test_is_done_all_completed():
    orch = Orchestrator(goal="g")
    orch.current_plan = _plan(["completed", "completed"])
    assert orch._is_done() is True


async def test_is_done_has_failure():
    orch = Orchestrator(goal="g")
    orch.current_plan = _plan(["completed", "failed"])
    assert orch._is_done() is False


async def test_is_done_incomplete():
    orch = Orchestrator(goal="g")
    orch.current_plan = _plan(["completed", "pending"])
    assert orch._is_done() is False


async def test_is_done_no_plan():
    orch = Orchestrator(goal="g")
    assert orch._is_done() is True


async def test_max_cycles_enforced(monkeypatch):
    monkeypatch.setattr(settings, "max_cycles", 1)
    monkeypatch.setattr(settings, "max_parallel_tasks", 5)

    plan = Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok")
    test_result = TestResult(passed=True, summary="ok", failures=[])

    worker_inst = mock.MagicMock()
    worker_inst.run = mock.AsyncMock(side_effect=RuntimeError("boom"))
    worker_cls = mock.MagicMock()
    worker_cls.return_value = worker_inst

    with mock.patch.object(orch_module, "PlannerAgent") as PlannerMock, \
            mock.patch.object(orch_module, "WorkerAgent", worker_cls), \
            mock.patch.object(orch_module, "TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = mock.AsyncMock(return_value=plan)
        TesterMock.return_value.run = mock.AsyncMock(return_value=test_result)
        orch = Orchestrator(goal="fix tests", client=_make_client())
        await orch.run()

    assert orch.cycles == 1
    assert orch.current_plan is plan


async def test_max_cycles_zero_runs_until_done(monkeypatch):
    monkeypatch.setattr(settings, "max_cycles", 0)
    monkeypatch.setattr(settings, "max_parallel_tasks", 5)

    plan = Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok")
    test_result = TestResult(passed=True, summary="ok", failures=[])

    with mock.patch.object(orch_module, "PlannerAgent") as PlannerMock, \
            mock.patch.object(orch_module, "WorkerAgent") as WorkerMock, \
            mock.patch.object(orch_module, "TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = mock.AsyncMock(return_value=plan)
        WorkerMock.return_value.run = mock.AsyncMock(return_value="done")
        TesterMock.return_value.run = mock.AsyncMock(return_value=test_result)
        orch = Orchestrator(goal="g", client=_make_client())
        await orch.run()

    assert orch.cycles == 1


async def test_semaphore_limits_concurrency(monkeypatch):
    monkeypatch.setattr(settings, "max_parallel_tasks", 2)
    monkeypatch.setattr(settings, "max_cycles", 0)

    state = {"current": 0, "peak": 0}

    class FakeWorker:
        def __init__(self, task: TaskModel, client=None, **kwargs) -> None:
            self.task = task

        async def run(self) -> str:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
            await asyncio.sleep(0.05)
            state["current"] -= 1
            return f"done-{self.task.id}"

    plan = Plan(
        tasks=[TaskModel(id=str(i), description=f"t{i}") for i in range(3)],
        rationale="ok",
    )
    test_result = TestResult(passed=True, summary="ok", failures=[])

    with mock.patch.object(orch_module, "PlannerAgent") as PlannerMock, \
            mock.patch.object(orch_module, "WorkerAgent", FakeWorker), \
            mock.patch.object(orch_module, "TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = mock.AsyncMock(return_value=plan)
        TesterMock.return_value.run = mock.AsyncMock(return_value=test_result)
        orch = Orchestrator(goal="g", client=_make_client())
        await orch.run()

    assert state["peak"] == 2


async def test_reporter_called_with_nonempty_messages(monkeypatch):
    monkeypatch.setattr(settings, "max_cycles", 1)
    monkeypatch.setattr(settings, "max_parallel_tasks", 5)

    reporter = mock.AsyncMock()
    plan = Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok")
    test_result = TestResult(passed=True, summary="ok", failures=[])

    with mock.patch.object(orch_module, "PlannerAgent") as PlannerMock, \
            mock.patch.object(orch_module, "WorkerAgent") as WorkerMock, \
            mock.patch.object(orch_module, "TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = mock.AsyncMock(return_value=plan)
        WorkerMock.return_value.run = mock.AsyncMock(return_value="work done")
        TesterMock.return_value.run = mock.AsyncMock(return_value=test_result)
        orch = Orchestrator(goal="g", client=_make_client(), reporter=reporter)
        await orch.run()

    assert reporter.await_count >= 1
    for call in reporter.call_args_list:
        msg = call.args[0]
        assert isinstance(msg, str)
        assert len(msg) > 0


async def test_reporter_failure_does_not_crash(monkeypatch):
    monkeypatch.setattr(settings, "max_cycles", 1)
    monkeypatch.setattr(settings, "max_parallel_tasks", 5)

    async def bad_reporter(msg: str) -> None:
        raise RuntimeError("reporter down")

    reporter = mock.AsyncMock(side_effect=bad_reporter)
    plan = Plan(tasks=[TaskModel(id="1", description="t1")], rationale="ok")
    test_result = TestResult(passed=True, summary="ok", failures=[])

    with mock.patch.object(orch_module, "PlannerAgent") as PlannerMock, \
            mock.patch.object(orch_module, "WorkerAgent") as WorkerMock, \
            mock.patch.object(orch_module, "TesterAgent") as TesterMock:
        PlannerMock.return_value.plan = mock.AsyncMock(return_value=plan)
        WorkerMock.return_value.run = mock.AsyncMock(return_value="work done")
        TesterMock.return_value.run = mock.AsyncMock(return_value=test_result)
        orch = Orchestrator(goal="g", client=_make_client(), reporter=reporter)
        await orch.run()

    assert orch.cycles == 1
    assert orch.original_goal == "g"
