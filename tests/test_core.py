import asyncio
import json
from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.agents.planner import PlannerAgent
from furrow.agents.tester import TesterAgent


class _FakePlannerClient:
    """Tiny deterministic fake LLM client for PlannerAgent tests."""

    def __init__(self, responses, settings=None):
        self._responses = list(responses)
        self.settings = settings or Settings(max_attempts=2)
        self.calls = 0

    async def complete(self, prompt, model=None):
        self.calls += 1
        if not self._responses:
            raise RuntimeError("No more canned responses")
        return self._responses.pop(0)


@contextmanager
def _patch_orchestrator(plan, test_result, worker_func=None):
    """Patch planner/worker/tester agents so run() never touches real LLMs or subprocesses."""

    async def fake_plan(self, goal):
        return plan

    async def fake_tester_run(self, goal, tasks):
        return test_result

    async def default_worker(self):
        return "done"

    with (
        patch("furrow.agents.planner.PlannerAgent.plan", new=fake_plan),
        patch("furrow.agents.worker.WorkerAgent.run", new=worker_func or default_worker),
        patch("furrow.agents.tester.TesterAgent.run", new=fake_tester_run),
    ):
        yield


def test_initial_is_done_false():
    orch = Orchestrator(goal="do something")
    assert orch._is_done() is False


def test_completed_tasks_require_passing_test():
    settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="do something", settings=settings)
    orch._has_work_started = True
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="completed"),
    ]
    orch._latest_test_passed = False
    assert orch._is_done() is False


def test_failed_worker_task_remains_incomplete():
    settings = Settings(max_cycles=1)
    orch = Orchestrator(goal="do something", settings=settings)
    orch._has_work_started = True
    orch.tasks = [
        TaskModel(id="1", description="task1", status="completed"),
        TaskModel(id="2", description="task2", status="failed"),
    ]
    orch._latest_test_passed = True
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_max_cycles_stops_loop():
    plan = Plan(tasks=[TaskModel(id="1", description="task1")], rationale="ok")
    test_result = TestResult(passed=False, summary="fail", failures=["x"])

    with _patch_orchestrator(plan, test_result):
        settings = Settings(max_cycles=2)
        orch = Orchestrator(goal="do something", settings=settings)
        await orch.run()

    assert orch.cycles == 2


@pytest.mark.asyncio
async def test_max_parallel_tasks_bounds_concurrency():
    tasks = [TaskModel(id=str(i), description=f"task{i}") for i in range(10)]
    plan = Plan(tasks=tasks, rationale="ok")
    test_result = TestResult(passed=True, summary="ok", failures=[])

    concurrency = 0
    max_concurrency = 0

    async def fake_worker_run(self):
        nonlocal concurrency, max_concurrency
        concurrency += 1
        max_concurrency = max(max_concurrency, concurrency)
        await asyncio.sleep(0.01)
        concurrency -= 1
        return "done"

    with _patch_orchestrator(plan, test_result, worker_func=fake_worker_run):
        settings = Settings(max_cycles=1, max_parallel_tasks=3)
        orch = Orchestrator(goal="do something", settings=settings)
        await orch.run()

    assert max_concurrency <= 3
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_original_goal_preserved_after_test_failure():
    plan = Plan(tasks=[TaskModel(id="1", description="task1")], rationale="ok")
    test_result = TestResult(passed=False, summary="fail", failures=["fix me"])

    with _patch_orchestrator(plan, test_result):
        settings = Settings(max_cycles=1)
        orch = Orchestrator(goal="original goal", settings=settings)
        await orch.run()

    assert orch.goal != "original goal"
    assert orch.original_goal == "original goal"


@pytest.mark.asyncio
async def test_empty_plan_with_failing_validation_does_not_complete():
    plan = Plan(tasks=[], rationale="nothing to do")
    test_result = TestResult(passed=False, summary="nope", failures=["still broken"])

    with _patch_orchestrator(plan, test_result):
        settings = Settings(max_cycles=2)
        orch = Orchestrator(goal="do something", settings=settings)
        await orch.run()

    assert orch._is_done() is False
    assert orch.cycles == 2
    assert orch._latest_test_passed is False


@pytest.mark.asyncio
async def test_empty_plan_with_passing_validation_completes():
    plan = Plan(tasks=[], rationale="nothing to do")
    test_result = TestResult(passed=True, summary="ok", failures=[])

    with _patch_orchestrator(plan, test_result):
        settings = Settings(max_cycles=5)
        orch = Orchestrator(goal="do something", settings=settings)
        await orch.run()

    assert orch._is_done() is True
    assert orch.cycles == 1
    assert orch._latest_test_passed is True


@pytest.mark.asyncio
async def test_worker_exception_becomes_failed_task_without_aborting_cycle():
    """A raising worker must mark the task failed and let the cycle finish testing."""
    plan = Plan(tasks=[TaskModel(id="1", description="boom")], rationale="ok")
    test_result = TestResult(passed=True, summary="ok", failures=[])

    async def fake_worker_run(self):
        raise RuntimeError("worker exploded")

    with _patch_orchestrator(plan, test_result, worker_func=fake_worker_run):
        settings = Settings(max_cycles=1)
        orch = Orchestrator(goal="do something", settings=settings)
        await orch.run()

    assert len(orch.tasks) == 1
    assert orch.tasks[0].status == "failed"
    assert orch.tasks[0].result is not None
    assert "worker exploded" in orch.tasks[0].result
    # The cycle did not abort: testing ran after the worker failure.
    assert orch._latest_test_passed is True
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_malformed_tester_output_is_failed_validation():
    """A non-JSON tester response must be treated as a failed validation."""
    fake_client = AsyncMock()
    fake_client.complete = AsyncMock(return_value="not valid json {{{")

    with patch.object(
        TesterAgent,
        "_run_tests",
        new=AsyncMock(return_value=("some output", False)),
    ):
        tester = TesterAgent(client=fake_client)
        result = await tester.run(goal="do something", tasks=[])

    assert result.passed is False
    assert result.summary == "Malformed tester response"
    assert any("not valid json" in f for f in result.failures)
    # The fake client was actually consulted with the test output.
    fake_client.complete.assert_awaited_once()


@pytest.mark.asyncio
async def test_planner_parses_fenced_json():
    """A Markdown fenced JSON response must parse into a Plan."""
    payload = json.dumps({"tasks": [{"id": "1", "description": "do thing"}], "rationale": "ok"})
    fenced = f"```json\n{payload}\n```"
    client = _FakePlannerClient([fenced], settings=Settings(max_attempts=2))

    planner = PlannerAgent(client=client)
    plan = await planner.plan("build a widget")

    assert plan.tasks[0].id == "1"
    assert plan.rationale == "ok"
    assert client.calls == 1


@pytest.mark.asyncio
async def test_planner_retries_after_malformed_then_succeeds():
    """A malformed response followed by a valid one must retry once and succeed."""
    valid = json.dumps({"tasks": [{"id": "2", "description": "task2"}], "rationale": "fine"})
    client = _FakePlannerClient(["not valid json {{{", valid], settings=Settings(max_attempts=2))

    planner = PlannerAgent(client=client)
    plan = await planner.plan("recover from bad output")

    assert plan.tasks[0].id == "2"
    assert client.calls == 2


@pytest.mark.asyncio
async def test_planner_exhausts_attempts_and_raises():
    """Exhausted malformed responses must raise and call the client max_attempts times."""
    client = _FakePlannerClient(["still bad", "nope"], settings=Settings(max_attempts=2))

    planner = PlannerAgent(client=client)
    with pytest.raises(ValueError) as exc_info:
        await planner.plan("never valid")

    assert "2 attempts" in str(exc_info.value)
    assert "nope" in str(exc_info.value)
    assert client.calls == 2