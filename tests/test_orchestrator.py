import asyncio
from typing import Any

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class DummySettings:
    def __init__(self, max_parallel_tasks: int = 2, max_cycles: int = 0):
        self.max_parallel_tasks = max_parallel_tasks
        self.max_cycles = max_cycles
        self.planner_model = "planner"
        self.worker_model = "worker"
        self.tester_model = "tester"


class DummyClient:
    def __init__(self, settings: DummySettings | None = None):
        self.settings = settings or DummySettings()


def make_orch(goal: str = "do it", settings: DummySettings | None = None) -> Orchestrator:
    client = DummyClient(settings)
    return Orchestrator(goal, client=client)


@pytest.mark.asyncio
async def test_run_stops_when_planner_returns_no_tasks(monkeypatch):
    orch = make_orch("goal")

    async def plan_empty(*args: Any, **kwargs: Any) -> Plan:
        return Plan(tasks=[], rationale="done")

    async def tester_run(*args: Any, **kwargs: Any) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_empty)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    assert orch._is_done()


@pytest.mark.asyncio
async def test_runs_multiple_cycles_then_stops(monkeypatch):
    orch = make_orch("goal")

    plans = [
        Plan(tasks=[TaskModel(id="t1", description="first")], rationale="r1"),
        Plan(tasks=[], rationale="r2"),
    ]
    call_count = {"p": 0}

    async def plan_seq(*args: Any, **kwargs: Any) -> Plan:
        idx = call_count["p"]
        call_count["p"] += 1
        return plans[idx] if idx < len(plans) else Plan(tasks=[], rationale="done")

    async def worker_run(_self: Any, task: TaskModel | None = None, *args: Any, **kwargs: Any) -> str:
        if task is None and args:
            task = args[0]
        if task is None and hasattr(_self, "task"):
            task = _self.task
        return f"done {task.id}"

    async def tester_run(*args: Any, **kwargs: Any) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_seq)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", worker_run)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    assert call_count["p"] >= 2
    assert orch._is_done()


@pytest.mark.asyncio
async def test_max_cycles_bounds_the_loop(monkeypatch):
    orch = make_orch("goal", settings=DummySettings(max_parallel_tasks=2, max_cycles=2))

    async def plan_forever(*args: Any, **kwargs: Any) -> Plan:
        return Plan(tasks=[TaskModel(id="t1", description="task")], rationale="r")

    async def worker_run(_self: Any, task: TaskModel | None = None, *args: Any, **kwargs: Any) -> str:
        if task is None and args:
            task = args[0]
        if task is None and hasattr(_self, "task"):
            task = _self.task
        return f"done {task.id}"

    async def tester_run(*args: Any, **kwargs: Any) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_forever)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", worker_run)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_max_parallel_tasks_limits_concurrency(monkeypatch):
    peak = 0
    current = 0
    lock = asyncio.Lock()

    async def worker_run(_self: Any, task: TaskModel, *args: Any, **kwargs: Any) -> str:
        nonlocal peak, current
        async with lock:
            current += 1
            if current > peak:
                peak = current
        await asyncio.sleep(0)
        async with lock:
            current -= 1
        return f"done {task.id}"

    orch = make_orch("goal", settings=DummySettings(max_parallel_tasks=1, max_cycles=1))

    async def plan_once(*args: Any, **kwargs: Any) -> Plan:
        return Plan(
            tasks=[
                TaskModel(id="t1", description="a"),
                TaskModel(id="t2", description="b"),
            ],
            rationale="r",
        )

    async def tester_run(*args: Any, **kwargs: Any) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_once)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", worker_run)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    assert peak <= 1


@pytest.mark.asyncio
async def test_failed_task_is_retried_without_duplicate_entries(monkeypatch):
    orch = make_orch("goal")

    plans = [
        Plan(tasks=[TaskModel(id="t1", description="fix")], rationale="r1"),
        Plan(tasks=[TaskModel(id="t1", description="fix")], rationale="r2"),
    ]
    call_count = {"p": 0}
    attempt_count = {"t1": 0}

    async def plan_seq(*args: Any, **kwargs: Any) -> Plan:
        idx = call_count["p"]
        call_count["p"] += 1
        return plans[idx] if idx < len(plans) else Plan(tasks=[], rationale="done")

    async def worker_run(_self: Any, task: TaskModel | None = None, *args: Any, **kwargs: Any) -> str:
        if task is None and args:
            task = args[0]
        if task is None and hasattr(_self, "task"):
            task = _self.task
        if task.id == "t1":
            attempt_count["t1"] += 1
            if attempt_count["t1"] == 1:
                raise ValueError("boom")
        return f"ok {task.id}"

    async def tester_run(*args: Any, **kwargs: Any) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_seq)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", worker_run)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    assert len([t for t in orch.tasks if t.id == "t1"]) == 1
    assert orch.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_original_goal_is_preserved_after_a_test_failure(monkeypatch):
    orch = make_orch("my original goal")

    async def plan_once(*args: Any, **kwargs: Any) -> Plan:
        return Plan(tasks=[TaskModel(id="t1", description="do")], rationale="r")

    async def worker_run(_self: Any, task: TaskModel | None = None, *args: Any, **kwargs: Any) -> str:
        if task is None and args:
            task = args[0]
        if task is None and hasattr(_self, "task"):
            task = _self.task
        return f"done {task.id}"

    async def tester_run(*args: Any, **kwargs: Any) -> TestResult:
        return TestResult(passed=False, summary="failed", failures=["fail1"])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_once)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", worker_run)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    assert orch.goal == "my original goal"


@pytest.mark.asyncio
async def test_on_event_emits_the_expected_event_sequence(monkeypatch):
    events: list[dict[str, Any]] = []
    orch = make_orch("goal")
    orch.on_event = lambda e: events.append(e)

    plans = [
        Plan(tasks=[TaskModel(id="t1", description="a")], rationale="r1"),
        Plan(tasks=[], rationale="r2"),
    ]
    call_count = {"p": 0}

    async def plan_seq(*args: Any, **kwargs: Any) -> Plan:
        idx = call_count["p"]
        call_count["p"] += 1
        return plans[idx] if idx < len(plans) else Plan(tasks=[], rationale="done")

    async def worker_run(_self: Any, task: TaskModel | None = None, *args: Any, **kwargs: Any) -> str:
        if task is None and args:
            task = args[0]
        if task is None and hasattr(_self, "task"):
            task = _self.task
        return f"done {task.id}"

    async def tester_run(*args: Any, **kwargs: Any) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_seq)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", worker_run)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    types = [e["type"] for e in events]
    assert types[0] == "cycle_start"
    assert "plan" in types
    assert "task" in types
    assert "test" in types
    assert types[-1] == "cycle_end"


@pytest.mark.asyncio
async def test_identical_plan_does_not_loop_forever(monkeypatch):
    orch = make_orch("goal")

    plan = Plan(tasks=[TaskModel(id="t1", description="same")], rationale="r")
    call_count = {"p": 0}

    async def plan_same(*args: Any, **kwargs: Any) -> Plan:
        call_count["p"] += 1
        if call_count["p"] > 5:
            return Plan(tasks=[], rationale="stop")
        return plan

    async def worker_run(_self: Any, task: TaskModel | None = None, *args: Any, **kwargs: Any) -> str:
        if task is None and args:
            task = args[0]
        if task is None and hasattr(_self, "task"):
            task = _self.task
        return f"done {task.id}"

    async def tester_run(*args: Any, **kwargs: Any) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_same)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", worker_run)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    assert call_count["p"] <= 2


@pytest.mark.asyncio
async def test_task_statuses_are_recorded(monkeypatch):
    orch = make_orch("goal")

    plans = [
        Plan(tasks=[TaskModel(id="t1", description="a")], rationale="r1"),
        Plan(tasks=[], rationale="r2"),
    ]
    call_count = {"p": 0}

    async def plan_seq(_: str) -> Plan:
        idx = call_count["p"]
        call_count["p"] += 1
        return plans[idx] if idx < len(plans) else Plan(tasks=[], rationale="done")

    async def worker_run(_self: Any, task: TaskModel) -> str:
        return f"done {task.id}"

    async def tester_run(_: str, __: Any) -> TestResult:
        return TestResult(passed=True, summary="ok", failures=[])

    monkeypatch.setattr("furrow.core.orchestrator.PlannerAgent.plan", plan_seq)
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent.run", worker_run)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent.run", tester_run)

    await orch.run()
    assert orch.tasks[0].status == "completed"