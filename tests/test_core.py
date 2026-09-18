import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core import orchestrator as orchestrator_module
from furrow.core.orchestrator import Orchestrator, RunStatus


class FakePlanner:
    def __init__(self, plans):
        self.plans = iter(plans)

    async def plan(self, goal):
        return next(self.plans)


class CompletedWorker:
    def __init__(self, task, client=None):
        self.task = task

    async def run(self):
        self.task.status = "completed"
        self.task.result = "done"
        return "done"


class FailedWorker:
    def __init__(self, task, client=None):
        self.task = task

    async def run(self):
        self.task.status = "failed"
        self.task.result = "worker failed"
        return "worker failed"


class FakeTester:
    def __init__(self, result):
        self.result = result

    async def run(self, goal, tasks):
        return self.result


def make_orchestrator(plans, max_cycles=1, client=None):
    orchestrator = Orchestrator(
        goal="goal",
        client=client or object(),
        settings=Settings(max_cycles=max_cycles),
    )
    orchestrator.planner = FakePlanner(plans)
    return orchestrator


def use_tester(monkeypatch, result):
    monkeypatch.setattr(
        orchestrator_module,
        "TesterAgent",
        lambda **kwargs: FakeTester(result),
    )


@pytest.mark.asyncio
async def test_successful_completion(monkeypatch):
    task = TaskModel(id="1", description="implement feature")
    plan = Plan(tasks=[task], rationale="one task")
    test_result = TestResult(passed=True, summary="all tests passed", failures=[])
    orchestrator = make_orchestrator([plan])
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", CompletedWorker)
    use_tester(monkeypatch, test_result)

    result = await orchestrator.run()

    assert result.status is RunStatus.SUCCESS
    assert result.success is True
    assert result.exit_code == 0
    assert result.cycles == 1
    assert result.test_result is test_result
    assert orchestrator.last_result == result
    assert orchestrator.last_test_result is test_result
    assert orchestrator.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_empty_plan_completes_without_workers_or_tester(monkeypatch):
    plan = Plan(tasks=[], rationale="already complete")
    orchestrator = make_orchestrator([plan])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("empty plans should not invoke collaborators")

    monkeypatch.setattr(orchestrator_module, "WorkerAgent", fail_if_called)
    monkeypatch.setattr(orchestrator_module, "TesterAgent", fail_if_called)

    result = await orchestrator.run()

    assert result.status is RunStatus.SUCCESS
    assert result.cycles == 1
    assert orchestrator.tasks == []
    assert orchestrator.last_test_result is None


@pytest.mark.asyncio
async def test_failed_test_is_not_reported_as_success(monkeypatch):
    task = TaskModel(id="1", description="implement feature")
    plan = Plan(tasks=[task], rationale="one task")
    test_result = TestResult(passed=False, summary="tests failed", failures=["boom"])
    orchestrator = make_orchestrator([plan], max_cycles=1)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", CompletedWorker)
    use_tester(monkeypatch, test_result)

    result = await orchestrator.run()

    assert result.status is RunStatus.CYCLE_LIMIT
    assert result.success is False
    assert result.exit_code == 2
    assert result.test_result is test_result
    assert orchestrator.last_test_result.passed is False


@pytest.mark.asyncio
async def test_failed_task_is_reported_as_failure(monkeypatch):
    task = TaskModel(id="1", description="implement feature")
    plan = Plan(tasks=[task], rationale="one task")
    test_result = TestResult(passed=True, summary="all tests passed", failures=[])
    orchestrator = make_orchestrator([plan])
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", FailedWorker)
    use_tester(monkeypatch, test_result)

    result = await orchestrator.run()

    assert result.status is RunStatus.FAILURE
    assert result.success is False
    assert result.exit_code == 1
    assert "Worker task failures" in result.message
    assert orchestrator.tasks[0].status == "failed"


@pytest.mark.asyncio
async def test_cycle_limit_is_a_hard_bound(monkeypatch):
    plans = [
        Plan(tasks=[TaskModel(id=str(cycle), description=f"task {cycle}")], rationale="retry")
        for cycle in range(2)
    ]
    orchestrator = make_orchestrator(plans, max_cycles=2)
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", CompletedWorker)
    use_tester(
        monkeypatch,
        TestResult(passed=False, summary="still failing", failures=["boom"]),
    )

    result = await orchestrator.run()

    assert result.status is RunStatus.CYCLE_LIMIT
    assert result.cycles == 2
    assert result.success is False
    assert result.exit_code == 2


@pytest.mark.asyncio
async def test_collaborator_exceptions_return_failure(monkeypatch):
    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="task")
    orchestrator = make_orchestrator([plan])
    monkeypatch.setattr(orchestrator_module, "WorkerAgent", CompletedWorker)

    class FailingTester:
        async def run(self, goal, tasks):
            raise RuntimeError("tester unavailable")

    monkeypatch.setattr(orchestrator_module, "TesterAgent", lambda **kwargs: FailingTester())

    result = await orchestrator.run()

    assert result.status is RunStatus.FAILURE
    assert result.success is False
    assert result.exit_code == 1
    assert "Tester failed" in result.message
    assert orchestrator.last_test_result is not None


def test_settings_reject_negative_max_cycles():
    with pytest.raises(ValueError):
        Settings(max_cycles=-1)


def test_explicit_max_cycles_rejects_negative_values():
    with pytest.raises(ValueError):
        Orchestrator(goal="goal", client=object(), settings=Settings(), max_cycles=-1)
