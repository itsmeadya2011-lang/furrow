import pytest

from furrow.config import Plan, TaskModel
from furrow.config import TestResult as FurrowTestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    plan = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert plan.tasks[0].description == "do thing"


def test_test_result():
    result = FurrowTestResult(passed=True, summary="ok", failures=[])
    assert result.passed is True


def test_is_done_is_false_for_pending_tasks():
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [TaskModel(id="1", description="pending task")]

    assert orchestrator._is_done() is False


def test_is_done_is_true_for_completed_tasks():
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="first task", status="completed"),
        TaskModel(id="2", description="second task", status="completed"),
    ]

    assert orchestrator._is_done() is True


def test_is_done_is_false_for_failed_tasks():
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="first task", status="completed"),
        TaskModel(id="2", description="second task", status="failed"),
    ]

    assert orchestrator._is_done() is False


def test_is_done_is_true_for_an_empty_ledger():
    assert Orchestrator(goal="test goal")._is_done() is True


class _FakePlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = list(plans)
        self.goals: list[str] = []

    async def plan(self, goal: str) -> Plan:
        self.goals.append(goal)
        return self.plans.pop(0)


class _FakeWorker:
    def __init__(self, task: TaskModel, client=None) -> None:
        self.task = task
        self.client = client

    async def run(self) -> str:
        return f"result:{self.task.id}"


class _FailingWorker:
    def __init__(self, task: TaskModel, client=None) -> None:
        self.task = task
        self.client = client

    async def run(self) -> str:
        raise RuntimeError("worker failed")


class _FakeTester:
    def __init__(self, results: list[FurrowTestResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[str, list[TaskModel]]] = []

    async def run(self, goal: str, tasks: list[TaskModel]) -> FurrowTestResult:
        self.calls.append((goal, tasks))
        return self.results.pop(0)


def _install_fakes(monkeypatch, tester: _FakeTester) -> None:
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", _FakeWorker)
    monkeypatch.setattr(
        "furrow.core.orchestrator.TesterAgent",
        lambda *_args, **_kwargs: tester,
    )


@pytest.mark.asyncio
async def test_cycle_retains_plan_tasks_and_marks_them_completed(monkeypatch):
    planned_tasks = [
        TaskModel(id="1", description="do one"),
        TaskModel(id="2", description="do two"),
    ]
    plan = Plan(tasks=planned_tasks, rationale="ok")
    fake_planner = _FakePlanner([plan])
    fake_tester = _FakeTester([FurrowTestResult(passed=True, summary="all good", failures=[])])
    _install_fakes(monkeypatch, fake_tester)

    orchestrator = Orchestrator(goal="test goal")
    orchestrator.planner = fake_planner
    await orchestrator._cycle()

    assert fake_planner.goals == ["test goal"]
    assert orchestrator._get_tasks() == planned_tasks
    assert all(task.status == "completed" for task in planned_tasks)
    assert all(task.result == f"result:{task.id}" for task in planned_tasks)
    assert fake_tester.calls == [("test goal", planned_tasks)]
    assert orchestrator._latest_test_outcome is True
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_cycle_retains_failed_worker_task(monkeypatch):
    planned_task = TaskModel(id="1", description="do one")
    plan = Plan(tasks=[planned_task], rationale="ok")
    fake_planner = _FakePlanner([plan])
    fake_tester = _FakeTester([FurrowTestResult(passed=True, summary="all good", failures=[])])
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", _FailingWorker)
    monkeypatch.setattr(
        "furrow.core.orchestrator.TesterAgent",
        lambda *_args, **_kwargs: fake_tester,
    )

    orchestrator = Orchestrator(goal="test goal")
    orchestrator.planner = fake_planner
    await orchestrator._cycle()

    assert orchestrator._get_tasks() == [planned_task]
    assert planned_task.status == "failed"
    assert planned_task.result == "worker failed"
    assert orchestrator._latest_test_outcome is True
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_failed_test_result_prevents_completion(monkeypatch):
    planned_task = TaskModel(id="1", description="do one")
    plan = Plan(tasks=[planned_task], rationale="ok")
    fake_planner = _FakePlanner([plan])
    fake_tester = _FakeTester(
        [FurrowTestResult(passed=False, summary="tests failed", failures=["failure"])]
    )
    _install_fakes(monkeypatch, fake_tester)

    orchestrator = Orchestrator(goal="test goal")
    orchestrator.planner = fake_planner
    await orchestrator._cycle()

    assert planned_task.status == "completed"
    assert orchestrator._get_tasks() == [planned_task]
    assert orchestrator._latest_test_outcome is False
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_empty_plan_clears_stale_state_and_completes(monkeypatch):
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [TaskModel(id="stale", description="stale", status="completed")]
    orchestrator._latest_test_outcome = False
    fake_planner = _FakePlanner([Plan(tasks=[], rationale="nothing to do")])
    orchestrator.planner = fake_planner

    await orchestrator._cycle()

    assert orchestrator._get_tasks() == []
    assert orchestrator._latest_test_outcome is None
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_multi_cycle_replaces_ledger_with_current_plan(monkeypatch):
    first_tasks = [TaskModel(id="1", description="first task")]
    second_tasks = [TaskModel(id="2", description="second task")]
    fake_planner = _FakePlanner(
        [
            Plan(tasks=first_tasks, rationale="first plan"),
            Plan(tasks=second_tasks, rationale="second plan"),
        ]
    )
    fake_tester = _FakeTester(
        [
            FurrowTestResult(passed=False, summary="tests failed", failures=["first failure"]),
            FurrowTestResult(passed=True, summary="tests passed", failures=[]),
        ]
    )
    _install_fakes(monkeypatch, fake_tester)

    orchestrator = Orchestrator(goal="test goal")
    orchestrator.planner = fake_planner
    await orchestrator._cycle()

    assert orchestrator._get_tasks() == first_tasks
    assert orchestrator._is_done() is False

    await orchestrator._cycle()

    assert fake_planner.goals == ["test goal", "Fix failing tests:\nfirst failure"]
    assert orchestrator._get_tasks() == second_tasks
    assert all(task.status == "completed" for task in second_tasks)
    assert orchestrator._latest_test_outcome is True
    assert orchestrator._is_done() is True
