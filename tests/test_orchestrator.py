import furrow.core.orchestrator as orch_mod
from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient


class FakePlanner:
    def __init__(self, tasks):
        self.tasks = tasks

    async def plan(self, goal):
        return Plan(
            tasks=[TaskModel(**t.model_dump()) for t in self.tasks], rationale="r"
        )


class FakeWorker:
    def __init__(self, task, client=None):
        self.task = task

    async def run(self):
        return f"done {self.task.id}"


class AlwaysPassTester:
    def __init__(self, client=None):
        pass

    async def run(self, goal, tasks):
        return TestResult(passed=True, summary="all green", failures=[])


class FlakyTester:
    """Fails the first time, passes afterwards."""

    def __init__(self, client=None):
        self.n = 0

    async def run(self, goal, tasks):
        self.n += 1
        if self.n == 1:
            return TestResult(passed=False, summary="broken", failures=["broken thing"])
        return TestResult(passed=True, summary="fixed", failures=[])


def _make_settings():
    from furrow.config import Provider, Settings

    settings = Settings()
    settings.provider = Provider.ANTHROPIC
    return settings


async def test_orchestrator_single_cycle_complete(monkeypatch):
    settings = _make_settings()
    settings.max_cycles = 1
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="do the thing", client=client)

    # The orchestrator pre-builds self.planner in __init__, so replace it on
    # the instance rather than patching the class.
    orch.planner = FakePlanner([TaskModel(id="1", description="task one")])
    monkeypatch.setattr(orch_mod, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda **kw: AlwaysPassTester())

    await orch.run()

    assert orch.plan is not None
    assert len(orch._get_tasks()) == 1
    assert orch._get_tasks()[0].status == "completed"
    assert orch._last_test_passed is True
    assert orch._is_done() is True


async def test_orchestrator_retries_on_test_failure(monkeypatch):
    """Tests failing in cycle 1 must not terminate the loop."""
    settings = _make_settings()
    settings.max_cycles = 5
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="go", client=client)

    orch.planner = FakePlanner([TaskModel(id="1", description="task one")])
    tester = FlakyTester()
    monkeypatch.setattr(orch_mod, "WorkerAgent", FakeWorker)
    monkeypatch.setattr(orch_mod, "TesterAgent", lambda **kw: tester)

    await orch.run()

    assert tester.n == 2
    assert orch.cycles == 2
    assert orch._last_test_passed is True
    assert orch._is_done() is True


async def test_orchestrator_max_cycles_enforced(monkeypatch):
    settings = _make_settings()
    settings.max_cycles = 3
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="go", client=client)

    orch.planner = FakePlanner([TaskModel(id="1", description="task one")])
    monkeypatch.setattr(orch_mod, "WorkerAgent", FakeWorker)

    class AlwaysFailTester:
        def __init__(self, client=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(passed=False, summary="nope", failures=["x"])

    monkeypatch.setattr(orch_mod, "TesterAgent", lambda **kw: AlwaysFailTester())

    await orch.run()

    assert orch.cycles == 3
    assert orch._is_done() is False


def test_orchestrator_get_tasks_tracks_plan():
    settings = _make_settings()
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="go", client=client)

    # No plan yet -> no tasks.
    assert orch._get_tasks() == []

    orch.plan = Plan(
        tasks=[
            TaskModel(id="1", description="a"),
            TaskModel(id="2", description="b"),
        ],
        rationale="r",
    )
    assert len(orch._get_tasks()) == 2
    # _get_tasks returns a copy, not the live reference.
    got = orch._get_tasks()
    got.clear()
    assert len(orch._get_tasks()) == 2


def test_orchestrator_is_done_logic():
    settings = _make_settings()
    client = LLMClient(settings=settings)
    orch = Orchestrator(goal="go", client=client)

    orch.plan = Plan(tasks=[TaskModel(id="1", description="a")], rationale="r")
    # Pending task -> not done.
    assert orch._is_done() is False

    orch.plan.tasks[0].status = "completed"
    orch._last_test_passed = True
    assert orch._is_done() is True

    # Failed task -> not done even if tests passed.
    orch.plan.tasks.append(TaskModel(id="2", description="b"))
    orch.plan.tasks[1].status = "failed"
    assert orch._is_done() is False

    # All completed but tests failed -> not done.
    orch.plan.tasks = [TaskModel(id="1", description="a")]
    orch.plan.tasks[0].status = "completed"
    orch._last_test_passed = False
    assert orch._is_done() is False

    # No tasks -> done.
    orch.plan.tasks = []
    assert orch._is_done() is True
