from __future__ import annotations

import asyncio
import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class DeterministicCycleHooks:
    def __init__(self, *, with_tasks: bool = True, test_passed: bool = True) -> None:
        self.with_tasks = with_tasks
        self.test_passed = test_passed
        self.plan_calls = 0
        self.execute_calls = 0
        self.test_calls = 0

    async def plan(self, goal: str) -> Plan:
        self.plan_calls += 1
        if not self.with_tasks:
            return Plan(tasks=[], rationale="done")
        return Plan(
            tasks=[TaskModel(id="1", description="task", status="pending")],
            rationale="ok",
        )

    async def execute_tasks(self, tasks: list[TaskModel]) -> list[str]:
        self.execute_calls += 1
        for task in tasks:
            task.status = "completed"
            task.result = "ok"
        return ["ok"]

    async def test(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.test_calls += 1
        if self.test_passed:
            return TestResult(passed=True, summary="ok", failures=[])
        return TestResult(
            passed=False,
            summary=f"fail {self.test_calls}",
            failures=[f"failure {self.test_calls}"],
        )


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_completed_tasks_with_passed_test_is_done():
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_completed_tasks_with_failed_test_is_not_done():
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    orch.last_test_result = TestResult(passed=False, summary="fail", failures=["x"])
    assert orch._is_done() is False


def test_failed_task_is_not_done():
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orch._is_done() is False


def test_empty_tasks_is_done():
    orch = Orchestrator(goal="test")
    orch.tasks = []
    assert orch._is_done() is True


def test_empty_tasks_with_failed_last_test_result_is_not_done():
    orch = Orchestrator(goal="test")
    orch.tasks = []
    orch.last_test_result = TestResult(passed=False, summary="fail", failures=["x"])
    assert orch._is_done() is False


def test_get_tasks_exposes_persisted_tasks():
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="task 1"),
        TaskModel(id="2", description="task 2"),
    ]
    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].id == "1"
    assert tasks[1].id == "2"


def test_last_test_result_is_persisted():
    orch = Orchestrator(goal="test")
    result = TestResult(passed=False, summary="x", failures=["y"])
    orch.last_test_result = result
    assert orch.last_test_result is result


def test_should_stop_when_done():
    orch = Orchestrator(goal="test", config=Settings(max_cycles=0))
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
    ]
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._should_stop() is True


def test_should_stop_at_max_cycles_when_not_done():
    orch = Orchestrator(goal="test", config=Settings(max_cycles=3))
    orch.cycles = 3
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
    ]
    orch.last_test_result = TestResult(passed=False, summary="fail", failures=["x"])
    assert orch._should_stop() is True


def test_should_stop_not_when_active():
    orch = Orchestrator(goal="test", config=Settings(max_cycles=5))
    orch.cycles = 2
    orch.tasks = [
        TaskModel(id="1", description="task 1", status="pending"),
    ]
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._should_stop() is False


async def test_run_completes_with_empty_plan(monkeypatch: pytest.MonkeyPatch):
    hooks = DeterministicCycleHooks(with_tasks=False)
    orch = Orchestrator(goal="test", config=Settings(max_cycles=5))
    monkeypatch.setattr(orch, "_plan", hooks.plan)
    monkeypatch.setattr(orch, "_execute_tasks", hooks.execute_tasks)
    monkeypatch.setattr(orch, "_test", hooks.test)

    await orch.run()

    assert orch._is_done() is True
    assert orch.cycles == 1
    assert hooks.plan_calls == 1
    assert hooks.execute_calls == 0
    assert hooks.test_calls == 0


async def test_run_stops_at_max_cycles(monkeypatch: pytest.MonkeyPatch):
    hooks = DeterministicCycleHooks(test_passed=False)
    orch = Orchestrator(goal="test", config=Settings(max_cycles=2))
    monkeypatch.setattr(orch, "_plan", hooks.plan)
    monkeypatch.setattr(orch, "_execute_tasks", hooks.execute_tasks)
    monkeypatch.setattr(orch, "_test", hooks.test)

    await orch.run()

    assert orch.cycles == 2
    assert hooks.plan_calls == 2
    assert hooks.execute_calls == 2
    assert hooks.test_calls == 2
    assert all(task.status == "completed" for task in orch.tasks)
    assert orch.last_test_result is not None
    assert orch.last_test_result.passed is False


def test_empty_tasks_after_passed_test_is_done():
    orch = Orchestrator(goal="test")
    orch.tasks = []
    orch.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


class BaseExceptionHooks:
    def __init__(self) -> None:
        self.plan_calls = 0
        self.execute_calls = 0
        self.test_calls = 0

    async def plan(self, goal: str) -> Plan:
        self.plan_calls += 1
        return Plan(
            tasks=[TaskModel(id="1", description="task", status="pending")],
            rationale="ok",
        )

    async def execute_tasks(self, tasks: list[TaskModel]) -> list[str | BaseException]:
        self.execute_calls += 1
        return [asyncio.CancelledError("cancelled")]

    async def test(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.test_calls += 1
        return TestResult(passed=True, summary="ok", failures=[])


async def test_baseexception_result_is_failed(monkeypatch: pytest.MonkeyPatch):
    hooks = BaseExceptionHooks()
    orch = Orchestrator(goal="test", config=Settings(max_cycles=1))
    monkeypatch.setattr(orch, "_plan", hooks.plan)
    monkeypatch.setattr(orch, "_execute_tasks", hooks.execute_tasks)
    monkeypatch.setattr(orch, "_test", hooks.test)

    await orch.run()

    assert hooks.plan_calls == 1
    assert hooks.execute_calls == 1
    assert orch.tasks[0].status == "failed"
    result = orch.tasks[0].result
    assert result is not None
    assert "CancelledError" in result
    assert orch._get_tasks() is orch.tasks


async def test_cycle_persists_tasks_and_get_tasks_returns_same_list(
    monkeypatch: pytest.MonkeyPatch,
):
    hooks = DeterministicCycleHooks(with_tasks=True, test_passed=True)
    orch = Orchestrator(goal="test", config=Settings(max_cycles=1))
    monkeypatch.setattr(orch, "_plan", hooks.plan)
    monkeypatch.setattr(orch, "_execute_tasks", hooks.execute_tasks)
    monkeypatch.setattr(orch, "_test", hooks.test)

    await orch._cycle()

    assert len(orch.tasks) == 1
    assert orch._get_tasks() is orch.tasks


async def test_run_empty_plan_max_cycles_zero(monkeypatch: pytest.MonkeyPatch):
    hooks = DeterministicCycleHooks(with_tasks=False)
    orch = Orchestrator(goal="test", config=Settings(max_cycles=0))
    monkeypatch.setattr(orch, "_plan", hooks.plan)
    monkeypatch.setattr(orch, "_execute_tasks", hooks.execute_tasks)
    monkeypatch.setattr(orch, "_test", hooks.test)

    await orch.run()

    assert orch._is_done() is True
    assert orch.cycles == 1
    assert hooks.plan_calls == 1
    assert hooks.execute_calls == 0
    assert hooks.test_calls == 0


class SuccessfulRetryHooks:
    def __init__(self) -> None:
        self.plan_calls = 0
        self.execute_calls = 0
        self.test_calls = 0

    async def plan(self, goal: str) -> Plan:
        self.plan_calls += 1
        return Plan(
            tasks=[TaskModel(id="1", description="task", status="pending")],
            rationale="ok",
        )

    async def execute_tasks(self, tasks: list[TaskModel]) -> list[str]:
        self.execute_calls += 1
        for task in tasks:
            task.status = "completed"
            task.result = "ok"
        return ["ok"]

    async def test(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        self.test_calls += 1
        if self.test_calls == 1:
            return TestResult(passed=False, summary="fail", failures=["x"])
        return TestResult(passed=True, summary="ok", failures=[])


async def test_run_succeeds_after_failed_test(monkeypatch: pytest.MonkeyPatch):
    hooks = SuccessfulRetryHooks()
    orch = Orchestrator(goal="test", config=Settings(max_cycles=2))
    monkeypatch.setattr(orch, "_plan", hooks.plan)
    monkeypatch.setattr(orch, "_execute_tasks", hooks.execute_tasks)
    monkeypatch.setattr(orch, "_test", hooks.test)

    await orch.run()

    assert orch.cycles == 2
    assert hooks.plan_calls == 2
    assert hooks.execute_calls == 2
    assert hooks.test_calls == 2
    assert orch.last_test_result is not None
    assert orch.last_test_result.passed is True
