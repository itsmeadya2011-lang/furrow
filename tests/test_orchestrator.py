import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakePlanner:
    def __init__(self, plans: list[Plan]):
        self._plans = list(plans)
        self.calls = 0

    async def plan(self, goal: str) -> Plan:
        p = self._plans.pop(0) if self._plans else Plan(tasks=[], rationale="done")
        self.calls += 1
        return p


class FakeWorker:
    """Records task runs; can be configured to fail for specific task ids."""

    def __init__(self, task: TaskModel, client=None, settings=None):
        self.task = task
        self.fail_ids: set[str] = set()
        self.results: dict[str, str] = {}

    async def run(self) -> str:
        if self.task.id in self.fail_ids:
            raise RuntimeError(f"worker failed on {self.task.id}")
        return self.results.get(self.task.id, f"done {self.task.id}")


class FakeTester:
    def __init__(self, client=None, settings=None):
        self.client = client
        self.results: list[TestResult] = []
        self.calls = 0
        self.default = TestResult(passed=True, summary="ok", failures=[])

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        r = self.results[self.calls] if self.calls < len(self.results) else self.default
        self.calls += 1
        return r


def _make_orchestrator(
    plans: list[Plan],
    test_results: list[TestResult],
    max_cycles: int = 0,
    worker_fail_ids: set[str] | None = None,
):
    from furrow.core.orchestrator import Orchestrator

    planner = FakePlanner(plans)
    tester = FakeTester()
    tester.results = test_results

    def planner_factory(client):
        return planner

    def worker_factory(task, client):
        w = FakeWorker(task, client)
        w.fail_ids = worker_fail_ids or set()
        return w

    def tester_factory(client):
        return tester

    orch = Orchestrator(
        goal="test goal",
        max_cycles=max_cycles,
        planner_factory=planner_factory,
        worker_factory=worker_factory,
        tester_factory=tester_factory,
    )
    return orch, planner, tester


@pytest.mark.asyncio
async def test_orchestrator_completes_when_all_tasks_done():
    plans = [Plan(tasks=[TaskModel(id="1", description="t1")], rationale="r")]
    orch, _, _ = _make_orchestrator(plans, [TestResult(passed=True, summary="ok", failures=[])])
    await orch.run()
    assert orch.cycles == 1
    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "completed"


@pytest.mark.asyncio
async def test_orchestrator_stores_tasks_across_cycles():
    plans = [
        Plan(tasks=[TaskModel(id="1", description="t1")], rationale="r"),
        Plan(tasks=[TaskModel(id="2", description="t2")], rationale="r"),
    ]
    # Cycle 1 fails tests so orchestrator continues to cycle 2; cycle 2 passes.
    orch, _, _ = _make_orchestrator(plans, [
        TestResult(passed=False, summary="fail", failures=["err"]),
        TestResult(passed=True, summary="ok", failures=[]),
    ])
    await orch.run()
    assert len(orch.tasks) == 2
    assert all(t.status == "completed" for t in orch.tasks)
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_orchestrator_halts_on_max_cycles():
    plans = [Plan(tasks=[TaskModel(id="1", description="t1")], rationale="r")] * 5
    fail_result = TestResult(passed=False, summary="fail", failures=["err"])

    def always_fail_factory(client):
        t = FakeTester(client)
        t.default = fail_result
        return t

    planner = FakePlanner(plans)

    def planner_factory(client):
        return planner

    orch = Orchestrator(
        goal="test goal",
        max_cycles=3,
        planner_factory=planner_factory,
        worker_factory=lambda task, client: FakeWorker(task, client),
        tester_factory=always_fail_factory,
    )
    await orch.run()
    assert orch.cycles == 3


@pytest.mark.asyncio
async def test_orchestrator_emits_events():
    events = []

    async def on_event(event, data):
        events.append((event, data))

    plans = [Plan(tasks=[TaskModel(id="1", description="t1")], rationale="r")]
    orch, _, _ = _make_orchestrator(plans, [TestResult(passed=True, summary="ok", failures=[])])
    orch.on_event = on_event
    await orch.run()
    assert any(e[0] == "start" for e in events)
    assert any(e[0] == "done" for e in events)


@pytest.mark.asyncio
async def test_orchestrator_failed_task_prevents_done():
    plans = [Plan(tasks=[TaskModel(id="1", description="t1")], rationale="r")]
    orch, _, _ = _make_orchestrator(
        plans,
        [TestResult(passed=True, summary="ok", failures=[])],
        worker_fail_ids={"1"},
    )
    await orch.run()
    assert orch.tasks[0].status == "failed"
    assert orch._is_done() is False


def test_orchestrator_no_tasks_returns_not_done():
    from furrow.core.orchestrator import Orchestrator

    orch = Orchestrator(goal="test")
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_orchestrator_stalls_when_failures_persist_and_no_new_tasks():
    """When a task fails and the planner produces no new tasks, the orchestrator halts."""
    plans = [
        Plan(tasks=[TaskModel(id="1", description="t1")], rationale="r"),
        Plan(tasks=[], rationale="nothing left"),
    ]
    orch, _, _ = _make_orchestrator(
        plans,
        [TestResult(passed=True, summary="ok", failures=[])],
        worker_fail_ids={"1"},
    )
    await orch.run()
    assert orch.tasks[0].status == "failed"
    assert orch._stalled() is True


@pytest.mark.asyncio
async def test_orchestrator_test_failure_prevents_done():
    """When tests fail, the orchestrator should not halt (it tries to fix in next cycle)."""
    plans = [
        Plan(tasks=[TaskModel(id="1", description="t1")], rationale="r"),
        Plan(tasks=[TaskModel(id="2", description="t2")], rationale="r"),
    ]
    orch, _, _ = _make_orchestrator(
        plans,
        [
            TestResult(passed=False, summary="broken", failures=["test_x failed"]),
            TestResult(passed=True, summary="ok", failures=[]),
        ],
    )
    await orch.run()
    assert orch.cycles == 2
    assert orch.last_test_result is not None
    assert orch.last_test_result.passed
