from __future__ import annotations

import asyncio

import furrow.core.orchestrator as orch_mod
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def _settings(**overrides) -> Settings:
    base: dict = {
        "provider": Provider.OLLAMA,
        "ollama_base_url": "http://test:11434",
    }
    base.update(overrides)
    return Settings(**base)


class FakePlanner:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def plan(self, goal: str) -> Plan:
        return Plan(
            tasks=[
                TaskModel(id="1", description="task one"),
                TaskModel(id="2", description="task two"),
                TaskModel(id="3", description="task three"),
            ],
            rationale="ok",
        )


class _Tracker:
    def __init__(self) -> None:
        self.current = 0
        self.max = 0


class FakeWorker:
    tracker: _Tracker = _Tracker()

    def __init__(self, task: TaskModel, *args, **kwargs) -> None:
        self.task = task

    async def run(self) -> str:
        FakeWorker.tracker.current += 1
        FakeWorker.tracker.max = max(FakeWorker.tracker.max, FakeWorker.tracker.current)
        await asyncio.sleep(0.02)
        FakeWorker.tracker.current -= 1
        return f"done-{self.task.id}"


class FakeTester:
    result: TestResult = TestResult(passed=True, summary="ok", failures=[])

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def run(self, goal: str, tasks: list) -> TestResult:
        return FakeTester.result


def _install_fakes(monkeypatch) -> None:
    monkeypatch.setattr(orch_mod, "PlannerAgent", FakePlanner)
    monkeypatch.setattr(orch_mod, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", FakeTester)


def test_get_tasks_returns_tracked_tasks():
    orch = Orchestrator(goal="g", settings=_settings())
    tasks = [TaskModel(id="1", description="t", status="completed")]
    orch._tasks = tasks
    assert orch._get_tasks() is tasks
    assert orch._get_tasks() != []


def test_is_done_empty_plan_is_done():
    # No tasks means nothing is pending: vacuously complete (and no test run).
    orch = Orchestrator(goal="g", settings=_settings())
    orch._tasks = []
    orch._last_test_result = None
    assert orch._is_done() is True


def test_is_done_pending_tasks():
    orch = Orchestrator(goal="g", settings=_settings())
    orch._tasks = [TaskModel(id="1", description="t")]
    orch._last_test_result = None
    assert orch._is_done() is False


def test_is_done_completed_no_test_result():
    orch = Orchestrator(goal="g", settings=_settings())
    tasks = [TaskModel(id="1", description="a"), TaskModel(id="2", description="b")]
    for t in tasks:
        t.status = "completed"
    orch._tasks = tasks
    orch._last_test_result = None
    assert orch._is_done() is True


def test_is_done_completed_tests_passed():
    orch = Orchestrator(goal="g", settings=_settings())
    tasks = [TaskModel(id="1", description="t", status="completed")]
    orch._tasks = tasks
    orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_is_done_completed_tests_failed():
    orch = Orchestrator(goal="g", settings=_settings())
    orch._tasks = [TaskModel(id="1", description="t", status="completed")]
    orch._last_test_result = TestResult(passed=False, summary="bad", failures=["x"])
    assert orch._is_done() is False


def test_is_done_task_failed():
    orch = Orchestrator(goal="g", settings=_settings())
    orch._tasks = [TaskModel(id="1", description="t", status="failed")]
    orch._last_test_result = None
    assert orch._is_done() is False


async def test_max_cycles_respected(monkeypatch):
    _install_fakes(monkeypatch)
    FakeTester.result = TestResult(passed=False, summary="fail", failures=["broken"])
    orch = Orchestrator(goal="g", settings=_settings(max_cycles=2))
    await orch.run()
    assert orch.cycles == 2
    assert orch.max_cycles == 2


async def test_max_cycles_zero_is_infinite(monkeypatch):
    _install_fakes(monkeypatch)
    FakeTester.result = TestResult(passed=True, summary="ok", failures=[])
    orch = Orchestrator(goal="g", settings=_settings(max_cycles=0))
    await orch.run()
    # Tests pass -> completes in a single cycle and halts via _is_done.
    assert orch.cycles == 1


async def test_max_parallel_tasks_limit_enforced(monkeypatch):
    _install_fakes(monkeypatch)
    FakeTester.result = TestResult(passed=True, summary="ok", failures=[])
    FakeWorker.tracker = _Tracker()
    orch = Orchestrator(goal="g", settings=_settings(max_parallel_tasks=2, max_cycles=0))
    await orch.run()
    assert FakeWorker.tracker.max <= 2
    assert FakeWorker.tracker.max == 2


async def test_max_parallel_tasks_one(monkeypatch):
    _install_fakes(monkeypatch)
    FakeTester.result = TestResult(passed=True, summary="ok", failures=[])
    FakeWorker.tracker = _Tracker()
    orch = Orchestrator(goal="g", settings=_settings(max_parallel_tasks=1, max_cycles=0))
    await orch.run()
    assert FakeWorker.tracker.max == 1
