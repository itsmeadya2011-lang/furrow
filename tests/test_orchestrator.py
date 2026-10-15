"""Tests for the Orchestrator: state tracking, max_cycles, parallelism,
dependency ordering, retry logic, and completion detection."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from furrow.config import Plan, Provider, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator


def make_settings(**overrides):
    """Create Settings suitable for testing without real API keys."""
    defaults = {
        "provider": Provider.ANTHROPIC,
        "model": "test-model",
        "planner_model": "test-planner",
        "worker_model": "test-worker",
        "tester_model": "test-tester",
        "anthropic_api_key": "test-key",
        "max_parallel_tasks": 5,
        "max_cycles": 0,
    }
    defaults.update(overrides)
    return Settings(**defaults)


def make_mock_client():
    """Return a MagicMock that quacks like an LLMClient."""
    return MagicMock()


def patch_agents(plan, worker_result="done", test_result=None):
    """Return a tuple of context managers that patch plan, worker.run, tester.run."""
    if test_result is None:
        test_result = TestResult(passed=True, summary="all good")

    plan_cm = patch("furrow.agents.planner.PlannerAgent.plan", new=AsyncMock(return_value=plan))
    worker_cm = patch("furrow.agents.worker.WorkerAgent.run", new=AsyncMock(return_value=worker_result))
    tester_cm = patch("furrow.agents.tester.TesterAgent.run", new=AsyncMock(return_value=test_result))
    return plan_cm, worker_cm, tester_cm


# ---------------------------------------------------------------------------
# _get_tasks / _is_done
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_orchestrator_tracks_tasks():
    """Tasks from the plan are stored and retrievable via _get_tasks."""
    settings = make_settings(max_cycles=1)
    orch = Orchestrator(goal="test", client=make_mock_client(), settings=settings)

    plan = Plan(
        tasks=[
            TaskModel(id="1", description="task A", files=["a.py"]),
            TaskModel(id="2", description="task B", files=["b.py"]),
        ],
        rationale="test plan",
    )
    pass_result = TestResult(passed=True, summary="all good")

    plan_cm, worker_cm, tester_cm = patch_agents(plan, test_result=pass_result)
    with plan_cm, worker_cm, tester_cm:
        await orch.run()

    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert all(t.status == "completed" for t in tasks)


@pytest.mark.asyncio
async def test_is_done_all_completed():
    """_is_done returns True when every tracked task is completed and tests passed."""
    settings = make_settings(max_cycles=0)
    orch = Orchestrator(goal="g", settings=settings)
    orch.tasks = {
        "1": TaskModel(id="1", description="t1", status="completed"),
        "2": TaskModel(id="2", description="t2", status="completed"),
    }
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_with_failures():
    """_is_done returns False when any task has failed."""
    settings = make_settings(max_cycles=0)
    orch = Orchestrator(goal="g", settings=settings)
    orch.tasks = {
        "1": TaskModel(id="1", description="t1", status="completed"),
        "2": TaskModel(id="2", description="t2", status="failed"),
    }
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_no_tasks():
    """_is_done returns True when no tasks exist."""
    settings = make_settings(max_cycles=0)
    orch = Orchestrator(goal="g", settings=settings)
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_with_pending():
    """_is_done returns False when tasks are still pending."""
    settings = make_settings(max_cycles=0)
    orch = Orchestrator(goal="g", settings=settings)
    orch.tasks = {
        "1": TaskModel(id="1", description="t1", status="completed"),
        "2": TaskModel(id="2", description="t2", status="pending"),
    }
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_all_completed_but_tests_failed():
    """_is_done returns False when all tasks complete but tests failed."""
    settings = make_settings(max_cycles=0)
    orch = Orchestrator(goal="g", settings=settings)
    orch._last_test_passed = False
    orch.tasks = {
        "1": TaskModel(id="1", description="t1", status="completed"),
    }
    assert orch._is_done() is False


# ---------------------------------------------------------------------------
# max_cycles
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_cycles_enforced():
    """Orchestrator stops after max_cycles even if tests keep failing."""
    settings = make_settings(max_cycles=1)
    orch = Orchestrator(goal="test", settings=settings)

    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    fail_result = TestResult(passed=False, summary="failing", failures=["error"])

    plan_cm, worker_cm, tester_cm = patch_agents(plan, test_result=fail_result)
    with plan_cm, worker_cm, tester_cm:
        await orch.run()

    # Cycle 1 executed (tests failed → not done), cycle 2 hit max_cycles check
    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_max_cycles_zero_unlimited_runs_until_done():
    """max_cycles=0 means unlimited; loop stops when tests pass."""
    settings = make_settings(max_cycles=0)
    orch = Orchestrator(goal="test", settings=settings)

    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    pass_result = TestResult(passed=True, summary="ok")

    plan_cm, worker_cm, tester_cm = patch_agents(plan, test_result=pass_result)
    with plan_cm, worker_cm, tester_cm:
        await orch.run()

    assert orch.cycles == 1


# ---------------------------------------------------------------------------
# Dependency ordering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_dependency_ordering():
    """Tasks with dependencies execute after their dependencies complete."""
    settings = make_settings(max_cycles=1, max_parallel_tasks=2)
    orch = Orchestrator(goal="test", settings=settings)

    execution_order = []
    plan = Plan(
        tasks=[
            TaskModel(id="dep", description="dependency task"),
            TaskModel(id="main", description="main task", dependencies=["dep"]),
        ],
        rationale="test plan",
    )
    pass_result = TestResult(passed=True, summary="ok")

    async def mock_run(self):
        execution_order.append(self.task.id)
        return "done"

    with patch("furrow.agents.planner.PlannerAgent.plan", new=AsyncMock(return_value=plan)), \
         patch("furrow.agents.worker.WorkerAgent.run", new=mock_run), \
         patch("furrow.agents.tester.TesterAgent.run", new=AsyncMock(return_value=pass_result)):
        await orch.run()

    assert execution_order.index("dep") < execution_order.index("main")


@pytest.mark.asyncio
async def test_no_dependencies_run_in_parallel():
    """Independent tasks execute concurrently within the parallelism limit."""
    settings = make_settings(max_cycles=1, max_parallel_tasks=5)
    orch = Orchestrator(goal="test", settings=settings)

    execution_spy = []
    plan = Plan(
        tasks=[
            TaskModel(id="a", description="task a"),
            TaskModel(id="b", description="task b"),
            TaskModel(id="c", description="task c"),
        ],
        rationale="test plan",
    )
    pass_result = TestResult(passed=True, summary="ok")

    async def mock_run(self):
        execution_spy.append(self.task.id)
        return "done"

    with patch("furrow.agents.planner.PlannerAgent.plan", new=AsyncMock(return_value=plan)), \
         patch("furrow.agents.worker.WorkerAgent.run", new=mock_run), \
         patch("furrow.agents.tester.TesterAgent.run", new=AsyncMock(return_value=pass_result)):
        await orch.run()

    assert len(execution_spy) == 3
    assert set(execution_spy) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# max_parallel_tasks
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_max_parallel_tasks_limit():
    """Only max_parallel_tasks run concurrently."""
    settings = make_settings(max_cycles=1, max_parallel_tasks=1)
    orch = Orchestrator(goal="test", settings=settings)

    concurrent = []
    max_concurrent = [0]
    plan = Plan(
        tasks=[TaskModel(id=str(i), description=f"task {i}") for i in range(5)],
        rationale="test plan",
    )
    pass_result = TestResult(passed=True, summary="ok")

    async def mock_run(self):
        concurrent.append(self.task.id)
        max_concurrent[0] = max(max_concurrent[0], len(concurrent))
        await asyncio.sleep(0)  # yield to allow interleaving
        concurrent.remove(self.task.id)
        return "done"

    with patch("furrow.agents.planner.PlannerAgent.plan", new=AsyncMock(return_value=plan)), \
         patch("furrow.agents.worker.WorkerAgent.run", new=mock_run), \
         patch("furrow.agents.tester.TesterAgent.run", new=AsyncMock(return_value=pass_result)):
        await orch.run()

    assert max_concurrent[0] == 1  # At most 1 concurrent task


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_retry_on_failure():
    """A task that fails once is retried and succeeds on the second attempt."""
    settings = make_settings(max_cycles=1)
    orch = Orchestrator(goal="test", settings=settings)
    orch._retry_limit = 3

    call_count = {"1": 0}
    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    pass_result = TestResult(passed=True, summary="ok")

    async def mock_run(self):
        call_count[self.task.id] += 1
        if call_count[self.task.id] < 2:
            raise RuntimeError("temporary failure")
        return "success"

    with patch("furrow.agents.planner.PlannerAgent.plan", new=AsyncMock(return_value=plan)), \
         patch("furrow.agents.worker.WorkerAgent.run", new=mock_run), \
         patch("furrow.agents.tester.TesterAgent.run", new=AsyncMock(return_value=pass_result)):
        await orch.run()

    assert call_count["1"] == 2
    assert orch.tasks["1"].status == "completed"


@pytest.mark.asyncio
async def test_task_fails_after_retries():
    """A task that always fails is marked as 'failed' after exhausting retries."""
    settings = make_settings(max_cycles=2)
    orch = Orchestrator(goal="test", settings=settings)
    orch._retry_limit = 2

    call_count = {"1": 0}
    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    pass_result = TestResult(passed=True, summary="ok")

    async def mock_run(self):
        call_count[self.task.id] += 1
        raise RuntimeError("persistent failure")

    with patch("furrow.agents.planner.PlannerAgent.plan", new=AsyncMock(return_value=plan)), \
         patch("furrow.agents.worker.WorkerAgent.run", new=mock_run), \
         patch("furrow.agents.tester.TesterAgent.run", new=AsyncMock(return_value=pass_result)):
        await orch.run()

    assert call_count["1"] == 2
    assert orch.tasks["1"].status == "failed"


# ---------------------------------------------------------------------------
# Output callback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_output_callback_receives_messages():
    """The output_callback receives plain-text messages during execution."""
    settings = make_settings(max_cycles=1)
    received = []

    def callback(message: str) -> None:
        received.append(message)

    orch = Orchestrator(goal="test", settings=settings, output_callback=callback)

    plan = Plan(tasks=[TaskModel(id="1", description="task")], rationale="test")
    pass_result = TestResult(passed=True, summary="ok")

    plan_cm, worker_cm, tester_cm = patch_agents(plan, test_result=pass_result)
    with plan_cm, worker_cm, tester_cm:
        await orch.run()

    assert len(received) > 0
    assert all(isinstance(msg, str) and len(msg) > 0 for msg in received)
    # Plain-text console should strip Rich markup tags (e.g. [bold green])
    assert all("[/" not in msg and "[bold" not in msg for msg in received)
