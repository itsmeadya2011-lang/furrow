import pytest
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def test_plan_parse():
    p = Plan(tasks=[TaskModel(id="1", description="do thing")], rationale="ok")
    assert p.tasks[0].description == "do thing"


def test_test_result():
    t = TestResult(passed=True, summary="ok", failures=[])
    assert t.passed is True


def test_is_done_false_for_pending_tasks():
    """_is_done() should be false when tasks are pending."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="task 1", status="pending"),
        TaskModel(id="2", description="task 2", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_false_when_any_task_failed():
    """_is_done() should be false when any task has failed."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="failed"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_true_only_when_all_completed():
    """_is_done() should be true only when all tasks are completed."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="1", description="task 1", status="completed"),
        TaskModel(id="2", description="task 2", status="completed"),
    ]
    assert orchestrator._is_done() is True


class _FakePlanner:
    def __init__(self, plan: Plan) -> None:
        self._plan = plan
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        self.calls += 1
        return self._plan


class _FakeWorker:
    def __init__(self, task: TaskModel, client=None) -> None:
        self.task = task
        self.client = client

    async def run(self) -> str:
        return f"result:{self.task.id}"


class _FakeTester:
    def __init__(self, result: TestResult) -> None:
        self._result = result

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return self._result


@pytest.mark.asyncio
async def test_cycle_wires_planner_worker_tester_and_updates_status(monkeypatch):
    """_cycle() should run planner/worker/tester and reflect completed tasks."""
    planned_tasks = [
        TaskModel(id="1", description="do one"),
        TaskModel(id="2", description="do two"),
    ]
    plan = Plan(tasks=planned_tasks, rationale="ok")
    fake_planner = _FakePlanner(plan)
    fake_tester = _FakeTester(TestResult(passed=True, summary="all good", failures=[]))

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", _FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", lambda *a, **kw: fake_tester)

    orchestrator = Orchestrator(goal="test goal")
    orchestrator.planner = fake_planner
    await orchestrator._cycle()

    assert fake_planner.calls == 1
    assert orchestrator._get_tasks() is planned_tasks
    assert all(t.status == "completed" for t in orchestrator._get_tasks())
    assert all(t.result and t.result.startswith("result:") for t in orchestrator._get_tasks())


@pytest.mark.asyncio
async def test_cycle_failed_test_result_keeps_completed_tasks_from_completing(monkeypatch):
    """A failed test result must prevent completion after worker tasks succeed."""
    planned_tasks = [
        TaskModel(id="1", description="do one"),
    ]
    plan = Plan(tasks=planned_tasks, rationale="ok")
    fake_planner = _FakePlanner(plan)
    fake_tester = _FakeTester(TestResult(passed=False, summary="tests failed", failures=["failure"]))

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", _FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", lambda *a, **kw: fake_tester)

    orchestrator = Orchestrator(goal="test goal")
    orchestrator.planner = fake_planner
    await orchestrator._cycle()

    assert orchestrator._latest_test_outcome is False
    assert all(t.status == "completed" for t in orchestrator._get_tasks())
    assert orchestrator._is_done() is False


@pytest.mark.asyncio
async def test_cycle_passing_test_result_allows_completed_tasks_to_complete(monkeypatch):
    """A passing test result must allow completion after worker tasks succeed."""
    planned_tasks = [
        TaskModel(id="1", description="do one"),
    ]
    plan = Plan(tasks=planned_tasks, rationale="ok")
    fake_planner = _FakePlanner(plan)
    fake_tester = _FakeTester(TestResult(passed=True, summary="tests passed", failures=[]))

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", _FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", lambda *a, **kw: fake_tester)

    orchestrator = Orchestrator(goal="test goal")
    orchestrator.planner = fake_planner
    await orchestrator._cycle()

    assert orchestrator._latest_test_outcome is True
    assert all(t.status == "completed" for t in orchestrator._get_tasks())
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_empty_plan_after_failed_test_clears_completion_state(monkeypatch):
    """An empty plan must clear a stale failed test outcome and signal no work remains."""
    planned_tasks = [
        TaskModel(id="1", description="do one"),
    ]
    failed_plan = Plan(tasks=planned_tasks, rationale="fix me")
    fake_planner = _FakePlanner(failed_plan)
    fake_tester = _FakeTester(TestResult(passed=False, summary="tests failed", failures=["failure"]))

    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", _FakeWorker)
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", lambda *a, **kw: fake_tester)

    orchestrator = Orchestrator(goal="test goal")
    orchestrator.planner = fake_planner
    await orchestrator._cycle()
    assert orchestrator._is_done() is False

    empty_plan = Plan(tasks=[], rationale="nothing to do")
    orchestrator.planner = _FakePlanner(empty_plan)
    await orchestrator._cycle()

    assert orchestrator._latest_test_outcome is None
    assert orchestrator._get_tasks() == []
    assert orchestrator._is_done() is True


@pytest.mark.asyncio
async def test_empty_plan_clears_stale_tasks_and_is_complete(monkeypatch):
    """An empty plan should clear stale tasks and be considered complete."""
    orchestrator = Orchestrator(goal="test goal")
    orchestrator._tasks = [
        TaskModel(id="stale", description="leftover", status="completed"),
    ]
    assert orchestrator._is_done() is True

    empty_plan = Plan(tasks=[], rationale="nothing to do")
    fake_planner = _FakePlanner(empty_plan)

    orchestrator.planner = fake_planner
    await orchestrator._cycle()

    assert orchestrator._get_tasks() == []
    assert orchestrator._is_done() is True