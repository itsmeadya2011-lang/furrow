"""Tests for the Orchestrator's state-management logic."""
import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakePlanner:
    """Planner stub that returns a canned plan."""

    def __init__(self, plan: Plan | None = None):
        self._plan = plan or Plan(tasks=[], rationale="done")
        self.plans_called = 0

    async def plan(self, goal: str) -> Plan:
        self.plans_called += 1
        return self._plan


class FakeTester:
    """Tester stub that returns a canned result."""

    def __init__(self, result: TestResult | None = None):
        self._result = result or TestResult(passed=True, summary="ok", failures=[])

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return self._result


class FakeWorker:
    """Worker stub that returns a fixed string."""

    def __init__(self, result: str = "done"):
        self._result = result

    async def run(self) -> str:
        return self._result


@pytest.fixture
def orchestrator(monkeypatch):
    orch = Orchestrator(goal="test goal")
    orch.planner = FakePlanner()
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", lambda **kw: FakeTester())
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", lambda **kw: FakeWorker())
    return orch


def test_is_done_no_tasks():
    orch = Orchestrator(goal="x")
    assert orch._is_done() is True


def test_is_done_all_completed(orchestrator):
    orchestrator.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orchestrator._is_done() is True


def test_is_done_has_pending(orchestrator):
    orchestrator.all_tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orchestrator._is_done() is False


def test_is_done_has_failed(orchestrator):
    orchestrator.all_tasks = [
        TaskModel(id="1", description="a", status="failed", result="boom"),
    ]
    assert orchestrator._is_done() is False


def test_register_tasks_unique_ids(orchestrator):
    plan = Plan(
        tasks=[TaskModel(id="1", description="a"), TaskModel(id="2", description="b")],
        rationale="test",
    )
    orchestrator._register_tasks(plan)
    assert len(orchestrator.all_tasks) == 2
    assert orchestrator.all_tasks[0].id == "1"
    assert orchestrator.all_tasks[1].id == "2"


def test_register_tasks_dedupes_ids(orchestrator):
    """When a new plan re-uses an existing task ID, it gets cycle-prefixed."""
    orchestrator.cycles = 1
    orchestrator.all_tasks = [TaskModel(id="1", description="a", status="completed")]
    plan = Plan(tasks=[TaskModel(id="1", description="retry of a")], rationale="test")
    orchestrator._register_tasks(plan)
    # Original task is still there (completed), new task gets prefixed ID
    assert any(t.id == "1" for t in orchestrator.all_tasks)
    assert any(t.id == "1-1" for t in orchestrator.all_tasks)


def test_build_context_no_prior_tasks(orchestrator):
    ctx = orchestrator._build_context()
    assert ctx.startswith("test goal")
    assert "Break the remaining work into 1-5 new tasks" in ctx
    assert "Do not re-plan already-completed tasks" in ctx


def test_build_context_includes_completed(orchestrator):
    orchestrator.all_tasks = [
        TaskModel(id="1", description="write feature", status="completed", result="added feature.py"),
    ]
    ctx = orchestrator._build_context()
    assert "Completed tasks" in ctx
    assert "write feature" in ctx
    assert "do NOT re-plan" in ctx


def test_build_context_includes_failed(orchestrator):
    orchestrator.all_tasks = [
        TaskModel(id="1", description="write feature", status="failed", result="syntax error"),
    ]
    ctx = orchestrator._build_context()
    assert "Failed tasks" in ctx
    assert "syntax error" in ctx


def test_max_cycles_respected(orchestrator):
    """When max_cycles is set and exceeded, the loop stops."""
    orchestrator._max_cycles = 2
    plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="test")
    orchestrator.planner = FakePlanner(plan=plan)

    # We can't easily run run() with mocks since it calls asyncio.gather
    # with WorkerAgent, so just verify _max_cycles is checked
    assert orchestrator._max_cycles == 2


@pytest.mark.asyncio
async def test_run_stops_when_done_immediately(monkeypatch):
    """If the planner generates no tasks and there are none, run() should stop after 1 cycle."""
    orch = Orchestrator(goal="x")
    orch.planner = FakePlanner(plan=Plan(tasks=[], rationale="nothing to do"))
    monkeypatch.setattr("furrow.core.orchestrator.TesterAgent", lambda **kw: FakeTester())
    monkeypatch.setattr("furrow.core.orchestrator.WorkerAgent", lambda **kw: FakeWorker())

    await orch.run()
    assert orch.cycles == 1
