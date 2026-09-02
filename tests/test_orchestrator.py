from __future__ import annotations

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core import orchestrator as orch_mod
from furrow.core.orchestrator import Orchestrator


class StubClient:
    def __init__(self, max_cycles: int = 0) -> None:
        self.settings = type("S", (), {"max_cycles": max_cycles})()


class StubPlanner:
    def __init__(self, plans: list[Plan]) -> None:
        self.plans = plans
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        idx = self.calls if self.calls < len(self.plans) else len(self.plans) - 1
        self.calls += 1
        return self.plans[idx]


class StubWorker:
    def __init__(self, task: TaskModel, client: object = None) -> None:
        self.task = task

    async def run(self) -> str:
        self.task.status = "completed"
        self.task.result = f"out-{self.task.id}"
        return self.task.result


class StubTester:
    def __init__(self, client: object = None) -> None:
        self.client = client

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return self.result


@pytest.fixture
def make_orch(monkeypatch):
    def _make(plans: list[Plan], tester_result: TestResult, max_cycles: int = 0):
        monkeypatch.setattr(orch_mod, "WorkerAgent", StubWorker)
        monkeypatch.setattr(orch_mod, "TesterAgent", lambda client=None: StubTester(client))
        client = StubClient(max_cycles=max_cycles)
        orch = Orchestrator(goal="test-goal", client=client)
        orch.planner = StubPlanner(plans)
        stub_tester = StubTester(client=client)
        stub_tester.result = tester_result
        orch._stub_tester = stub_tester
        return orch

    return _make


def _patch_tester(monkeypatch, result: TestResult) -> None:
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client=None: _OneResultTester(result))


class _OneResultTester:
    def __init__(self, result: TestResult) -> None:
        self.result = result

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return self.result


def test_is_done_all_completed_and_tests_passed(monkeypatch) -> None:
    monkeypatch.setattr(orch_mod, "WorkerAgent", StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client=None: _OneResultTester(TestResult(passed=True, summary="ok")))
    orch = Orchestrator(goal="g", client=StubClient())
    orch.tasks = [TaskModel(id="1", description="x", status="completed")]
    orch.last_test_passed = True
    assert orch._is_done() is True


def test_is_done_false_when_task_failed(monkeypatch) -> None:
    monkeypatch.setattr(orch_mod, "WorkerAgent", StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client=None: _OneResultTester(TestResult(passed=True, summary="ok")))
    orch = Orchestrator(goal="g", client=StubClient())
    orch.tasks = [TaskModel(id="1", description="x", status="failed")]
    orch.last_test_passed = True
    assert orch._is_done() is False


def test_is_done_false_when_tests_failed(monkeypatch) -> None:
    monkeypatch.setattr(orch_mod, "WorkerAgent", StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client=None: _OneResultTester(TestResult(passed=False, summary="x")))
    orch = Orchestrator(goal="g", client=StubClient())
    orch.tasks = [TaskModel(id="1", description="x", status="completed")]
    orch.last_test_passed = False
    assert orch._is_done() is False


def test_is_done_empty_tasks(monkeypatch) -> None:
    monkeypatch.setattr(orch_mod, "WorkerAgent", StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client=None: _OneResultTester(TestResult(passed=True, summary="ok")))
    orch = Orchestrator(goal="g", client=StubClient())

    orch.tasks = []
    orch.last_test_passed = True
    assert orch._is_done() is True

    orch.last_test_passed = False
    assert orch._is_done() is False

    orch.tasks = [TaskModel(id="1", description="x", status="completed")]
    orch.last_test_passed = True
    assert orch._is_done() is True

    orch.last_test_passed = False
    assert orch._is_done() is False


async def test_run_halts_after_one_successful_cycle(monkeypatch) -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
    tester_result = TestResult(passed=True, summary="ok", failures=[])
    monkeypatch.setattr(orch_mod, "WorkerAgent", StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client=None: _OneResultTester(tester_result))
    client = StubClient(max_cycles=0)
    orch = Orchestrator(goal="g", client=client)
    orch.planner = StubPlanner([plan])

    await orch.run()

    assert orch.cycles == 1
    assert orch.last_test_passed is True


async def test_run_respects_max_cycles(monkeypatch) -> None:
    plan = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r")
    tester_result = TestResult(passed=False, summary="x", failures=["y"])
    monkeypatch.setattr(orch_mod, "WorkerAgent", StubWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda client=None: _OneResultTester(tester_result))
    client = StubClient(max_cycles=1)
    orch = Orchestrator(goal="g", client=client)
    orch.planner = StubPlanner([plan])

    await orch.run()

    assert orch.cycles == 1
    assert orch.last_test_passed is False


async def test_run_loops_until_done(monkeypatch) -> None:
    """Tests fail on cycle 1, pass on cycle 2. Verifies the loop iterates
    until _is_done returns True."""
    plan1 = Plan(tasks=[TaskModel(id="1", description="x")], rationale="r1")
    plan2 = Plan(tasks=[TaskModel(id="2", description="y")], rationale="r2"])
    fail_result = TestResult(passed=False, summary="fail", failures=["boom"])
    pass_result = TestResult(passed=True, summary="ok", failures=[])
    monkeypatch.setattr(orch_mod, "WorkerAgent", StubWorker)

    # Tester returns fail first, then pass on second call.
    tester_states = [fail_result, pass_result]

    def tester_factory(client=None):
        return _SequenceTester(tester_states)

    monkeypatch.setattr(orch_mod, "TesterAgent", tester_factory)
    client = StubClient(max_cycles=0)
    orch = Orchestrator(goal="g", client=client)
    orch.planner = StubPlanner([plan1, plan2])

    await orch.run()

    assert orch.cycles == 2
    assert orch.planner.calls == 2
    assert orch.last_test_passed is True


class _SequenceTester:
    def __init__(self, results):
        self.results = results
        self.idx = 0

    async def run(self, goal, tasks):
        result = self.results[min(self.idx, len(self.results) - 1)]
        self.idx += 1
        return result