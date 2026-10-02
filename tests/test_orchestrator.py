"""Tests for the Orchestrator."""

import asyncio

import pytest

from furrow.config import Plan, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


class FakePlanner:
    """Fake planner that returns a pre-configured plan."""

    def __init__(self, plan: Plan) -> None:
        self._plan = plan

    async def plan(self, goal: str) -> Plan:
        return self._plan


class FakeWorker:
    """Fake worker that returns a configurable result."""

    def __init__(self, result: str = "done") -> None:
        self.result = result

    async def run(self) -> str:
        return self.result


class FakeTester:
    """Fake tester that returns a pre-configured test result."""

    def __init__(self, result: TestResult) -> None:
        self.result = result

    async def run(self, goal: str, tasks: list[TaskModel]) -> TestResult:
        return self.result


def make_plan(tasks: list[TaskModel]) -> Plan:
    return Plan(tasks=tasks, rationale="test plan")


@pytest.fixture
def passing_tasks() -> list[TaskModel]:
    return [
        TaskModel(id="1", description="Task A"),
        TaskModel(id="2", description="Task B"),
    ]


@pytest.fixture
def passing_test_result() -> TestResult:
    return TestResult(passed=True, summary="All passed", failures=[])


@pytest.fixture
def failing_test_result() -> TestResult:
    return TestResult(
        passed=False,
        summary="Tests failed",
        failures=["task_a.py:1: AssertionError"],
    )


@pytest.fixture
def basic_orchestrator(passing_tasks, passing_test_result):
    """Create an orchestrator with fake dependencies and a passing test."""
    orch = Orchestrator(goal="Do something")
    orch.planner = FakePlanner(make_plan(passing_tasks))
    orch.tester = FakeTester(passing_test_result)
    # Monkeypatch WorkerAgent to avoid real LLM calls
    import furrow.core.orchestrator as orch_module

    orch_module.WorkerAgent = lambda task, client: FakeWorker("completed")
    return orch


def test_orchestrator_init(basic_orchestrator):
    """Orchestrator initializes with goal and default settings."""
    assert basic_orchestrator.goal == "Do something"
    assert basic_orchestrator.cycles == 0
    assert basic_orchestrator.all_tasks == []
    assert basic_orchestrator._last_test_result is None


@pytest.mark.asyncio
async def test_orchestrator_completes_on_passing_test(basic_orchestrator):
    """Orchestrator terminates when all tasks complete and tests pass."""
    await basic_orchestrator.run()
    assert basic_orchestrator.all_tasks[0].status == "completed"
    assert basic_orchestrator._last_test_result.passed is True


@pytest.mark.asyncio
async def test_orchestrator_respects_max_cycles(passing_tasks, passing_test_result):
    """Orchestrator stops after max_cycles even if not done."""
    orch = Orchestrator(goal="Loop forever", max_cycles=2)
    # Make _is_done always return False to force cycle limit
    orch._is_done = lambda: False
    orch.planner = FakePlanner(make_plan(passing_tasks))
    orch.tester = FakeTester(passing_test_result)
    import furrow.core.orchestrator as orch_module

    orch_module.WorkerAgent = lambda task, client: FakeWorker("completed")

    await orch.run()
    assert orch.cycles == 2  # Should stop at cycle 2 due to max_cycles


@pytest.mark.asyncio
async def test_orchestrator_retries_on_failing_test(passing_tasks, failing_test_result):
    """Orchestrator continues when tests fail (until max_cycles)."""
    orch = Orchestrator(goal="Fix failing tests", max_cycles=2)
    orch.planner = FakePlanner(make_plan(passing_tasks))
    orch.tester = FakeTester(failing_test_result)
    # _is_done returns False because tests failed
    import furrow.core.orchestrator as orch_module

    orch_module.WorkerAgent = lambda task, client: FakeWorker("completed")

    await orch.run()
    assert orch.cycles == 2  # Tests never pass, hits max_cycles


@pytest.mark.asyncio
async def test_orchestrator_tracks_task_status(passing_tasks, passing_test_result):
    """Tasks have their status updated after execution."""
    orch = Orchestrator(goal="Test tracking")
    orch.planner = FakePlanner(make_plan(passing_tasks))
    orch.tester = FakeTester(passing_test_result)
    import furrow.core.orchestrator as orch_module

    orch_module.WorkerAgent = lambda task, client: FakeWorker("done")

    await orch.run()
    assert len(orch.all_tasks) == 2
    assert all(t.status == "completed" for t in orch.all_tasks)


@pytest.mark.asyncio
async def test_orchestrator_parallel_semaphore():
    """Tasks execute with max_parallel semaphore limiting concurrency."""
    import furrow.core.orchestrator as orch_module

    call_count = 0
    max_concurrent = 0

    class ConcurrentWorker:
        async def run(self) -> str:
            nonlocal call_count, max_concurrent
            call_count += 1
            max_concurrent = max(max_concurrent, call_count)
            await asyncio.sleep(0.05)
            call_count -= 1
            return "done"

    tasks = [TaskModel(id=str(i), description=f"Task {i}") for i in range(5)]
    plan = make_plan(tasks)

    orch = Orchestrator(goal="Parallel test", max_parallel=2, max_cycles=1)
    orch.planner = FakePlanner(plan)
    orch.tester = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    orch._is_done = lambda: True

    orch_module.WorkerAgent = lambda task, client: ConcurrentWorker()

    await orch.run()
    # With max_parallel=2, we should never see more than 2 concurrent
    assert max_concurrent <= 2, f"Expected max 2 concurrent, got {max_concurrent}"


@pytest.mark.asyncio
async def test_orchestrator_progress_callback():
    """Progress callback is invoked with status updates."""
    messages: list[str] = []

    tasks = [TaskModel(id="1", description="Do thing")]
    plan = make_plan(tasks)

    orch = Orchestrator(
        goal="Test callback",
        max_cycles=1,
        progress_callback=messages.append,
    )
    orch.planner = FakePlanner(plan)
    orch.tester = FakeTester(TestResult(passed=True, summary="ok", failures=[]))
    orch._is_done = lambda: True

    import furrow.core.orchestrator as orch_module

    orch_module.WorkerAgent = lambda task, client: FakeWorker("done")

    await orch.run()
    assert len(messages) > 0
    assert any("started" in m.lower() for m in messages)


def test_orchestrator_is_done_no_tasks():
    """_is_done returns False when no tasks have run."""
    orch = Orchestrator(goal="Test")
    assert orch._is_done() is False


def test_orchestrator_is_done_with_completed_tasks():
    """_is_done returns True when tasks completed and tests passed."""
    orch = Orchestrator(goal="Test")
    task = TaskModel(id="1", description="done")
    task.status = "completed"
    orch.all_tasks = [task]
    orch._last_test_result = TestResult(passed=True, summary="ok", failures=[])
    assert orch._is_done() is True


def test_orchestrator_is_done_with_failed_tasks():
    """_is_done returns False when any task failed."""
    orch = Orchestrator(goal="Test")
    task = TaskModel(id="1", description="fail")
    task.status = "failed"
    orch.all_tasks = [task]
    assert orch._is_done() is False


def test_orchestrator_is_done_tests_not_passed():
    """_is_done returns False when tasks completed but tests failed."""
    orch = Orchestrator(goal="Test")
    task = TaskModel(id="1", description="done")
    task.status = "completed"
    orch.all_tasks = [task]
    orch._last_test_result = TestResult(passed=False, summary="fail", failures=["err"])
    assert orch._is_done() is False
