import asyncio

from furrow.config import Plan, TaskModel, TestResult
from furrow.core import orchestrator as orch_mod
from furrow.core.orchestrator import Orchestrator


def _orch(goal: str = "demo") -> Orchestrator:
    # Construction is cheap: no network calls at init (LLM is only used on run).
    return Orchestrator(goal=goal)


def test_get_tasks_returns_stored_tasks():
    o = _orch()
    t1 = TaskModel(id="1", description="a")
    t2 = TaskModel(id="2", description="b")
    o.tasks = [t1, t2]
    assert o._get_tasks() == [t1, t2]


def test_is_done_true_when_no_tasks():
    o = _orch()
    o.tasks = []
    assert o._is_done() is True


def test_is_done_true_when_tests_passed():
    o = _orch()
    o.tasks = [TaskModel(id="1", description="a")]
    o.last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert o._is_done() is True


def test_is_done_false_when_tests_failed():
    o = _orch()
    o.tasks = [TaskModel(id="1", description="a")]
    o.last_test_result = TestResult(passed=False, summary="boom", failures=["f1"])
    assert o._is_done() is False


def test_is_done_false_when_no_test_result():
    o = _orch()
    o.tasks = [TaskModel(id="1", description="a")]
    assert o.last_test_result is None
    assert o._is_done() is False


def test_max_cycles_zero_means_unlimited():
    o = _orch()
    o.settings.max_cycles = 0
    o.cycles = 1000
    assert o._max_cycles_reached() is False


def test_max_cycles_enforced():
    o = _orch()
    o.settings.max_cycles = 2
    o.cycles = 2
    assert o._max_cycles_reached() is True
    o.cycles = 1
    assert o._max_cycles_reached() is False


def _install_fakes(monkeypatch, plan, test_result):
    class FakePlanner:
        async def plan(self, goal):
            return plan

    class FakeWorker:
        def __init__(self, task, client=None):
            pass

        async def run(self):
            return "ok"

    class FakeTester:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return test_result

    monkeypatch.setattr(orch_mod, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", FakeTester)
    return FakePlanner


def test_cycle_continues_on_test_failure(monkeypatch):
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
    failing = TestResult(passed=False, summary="boom", failures=["failure one"])
    FakePlanner = _install_fakes(monkeypatch, plan, failing)

    o = _orch(goal="make tests fail so we continue")
    o.planner = FakePlanner()
    asyncio.run(o._cycle())

    assert o.last_test_result.passed is False
    # Not done because tests failed -> the loop should continue with a fix goal.
    assert o._is_done() is False
    assert o.goal.startswith("Fix failing tests")
    assert "failure one" in o.goal


def test_cycle_marks_done_when_tests_pass(monkeypatch):
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
    passing = TestResult(passed=True, summary="all good", failures=[])
    FakePlanner = _install_fakes(monkeypatch, plan, passing)

    o = _orch(goal="make tests pass")
    o.planner = FakePlanner()
    asyncio.run(o._cycle())

    assert o.last_test_result.passed is True
    assert o._is_done() is True
