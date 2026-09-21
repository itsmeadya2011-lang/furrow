import asyncio
from collections.abc import Iterable

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator


class FakePlanner:
    def __init__(self, plans: Iterable[Plan]) -> None:
        self._plans = iter(plans)

    async def plan(self, goal: str) -> Plan:
        return next(self._plans)


class FakeTesterAgent:
    def __init__(
        self,
        client: object = None,
        results: Iterable[TestResult] = (),
        settings: object = None,
    ) -> None:
        self._results = iter(results)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return next(self._results)


def _tester(results: Iterable[TestResult]):
    return lambda **kwargs: FakeTesterAgent(results=results, **kwargs)


def _plan(*descriptions: str) -> Plan:
    return Plan(
        tasks=[TaskModel(id=str(index), description=description) for index, description in enumerate(descriptions)],
        rationale="test plan",
    )


def _orchestrator(planner: FakePlanner, settings: Settings) -> Orchestrator:
    orchestrator = Orchestrator("test goal", client=object(), settings=settings)
    orchestrator.planner = planner
    return orchestrator


def test_is_done_uses_persisted_task_state() -> None:
    orchestrator = Orchestrator("test goal", client=object(), settings=Settings())
    task = TaskModel(id="1", description="unfinished")
    orchestrator.tasks.append(task)

    assert orchestrator._get_tasks() == [task]
    assert orchestrator._is_done() is False

    task.status = "completed"
    orchestrator.latest_test_result = TestResult(passed=False, summary="failed", failures=["test failure"])
    assert orchestrator._is_done() is False

    orchestrator.latest_test_result = TestResult(passed=True, summary="passed", failures=[])
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_run_retries_failed_task_and_then_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: dict[str, int] = {}

    class FakeWorkerAgent:
        def __init__(self, task: TaskModel, client: object) -> None:
            self.task = task

        async def run(self) -> str:
            attempts[self.task.id] = attempts.get(self.task.id, 0) + 1
            if attempts[self.task.id] == 1:
                raise RuntimeError("worker failed")
            return "fixed"

    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FakeWorkerAgent)
    monkeypatch.setattr(
        orchestrator_module,
        "TesterAgent",
        FakeTesterAgent,
        raising=False,
    )
    orchestrator = _orchestrator(
        FakePlanner([_plan("retry me"), _plan("retry me")]),
        Settings(max_cycles=3),
    )
    orchestrator_module.TesterAgent = lambda **kwargs: FakeTesterAgent(
        results=[
            TestResult(passed=True, summary="passed", failures=[]),
            TestResult(passed=True, summary="passed", failures=[]),
        ],
        **kwargs,
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert attempts == {"0": 2}
    task = orchestrator._get_tasks()[0]
    assert task.status == "completed"
    assert task.result == "fixed"
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_run_stops_at_max_cycles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestrator_module,
        "WorkerAgent",
        lambda **kwargs: _ImmediateWorker("done"),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "TesterAgent",
        lambda **kwargs: _ImmediateTester(
            TestResult(passed=False, summary="failed", failures=["still failing"]),
            **kwargs,
        ),
    )
    orchestrator = _orchestrator(
        FakePlanner([_plan("pending"), _plan("pending")]),
        Settings(max_cycles=2),
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_zero_max_cycles_allows_later_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        orchestrator_module,
        "WorkerAgent",
        lambda **kwargs: _ImmediateWorker("done"),
    )
    monkeypatch.setattr(
        orchestrator_module,
        "TesterAgent",
        lambda **kwargs: _SequenceTester(
            [
                TestResult(passed=False, summary="failed", failures=["retry"]),
                TestResult(passed=True, summary="passed", failures=[]),
            ],
            **kwargs,
        ),
    )
    orchestrator = _orchestrator(
        FakePlanner([_plan("pending"), _plan("pending")]),
        Settings(max_cycles=0),
    )

    await orchestrator.run()

    assert orchestrator.cycles == 2
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_max_parallel_tasks_caps_worker_concurrency() -> None:
    settings = Settings(max_parallel_tasks=2)
    orchestrator = Orchestrator("test goal", client=object(), settings=settings)
    tasks = [TaskModel(id=str(index), description=f"task {index}") for index in range(5)]
    active = 0
    max_active = 0
    started = 0
    release_workers = asyncio.Event()

    class FakeWorkerAgent:
        def __init__(self, task: TaskModel, client: object) -> None:
            self.task = task

        async def run(self) -> str:
            nonlocal active, max_active, started
            active += 1
            started += 1
            max_active = max(max_active, active)
            if started == settings.max_parallel_tasks:
                release_workers.set()
            await asyncio.wait_for(release_workers.wait(), timeout=1)
            active -= 1
            return self.task.id

    monkeypatch_context = pytest.MonkeyPatch()
    monkeypatch_context.setattr(orchestrator_module, "WorkerAgent", FakeWorkerAgent)
    try:
        results = await orchestrator._execute_tasks(tasks)
    finally:
        monkeypatch_context.undo()

    assert results == [task.id for task in tasks]
    assert max_active == settings.max_parallel_tasks


class _ImmediateWorker:
    def __init__(self, result: str) -> None:
        self.result = result

    async def run(self) -> str:
        return self.result


class _ImmediateTester:
    def __init__(self, result: TestResult, **kwargs: object) -> None:
        self.result = result

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return self.result


class _SequenceTester:
    def __init__(self, results: list[TestResult], **kwargs: object) -> None:
        self._results = iter(results)

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return next(self._results)
