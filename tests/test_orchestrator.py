"""Unit tests for the Furrow orchestrator."""

import pytest

import furrow.core.orchestrator as orch_mod
from furrow.config import (
    Plan,
    Provider,
    Settings,
    TaskModel,
    TestResult,
    WorkerResult,
)
from furrow.core.orchestrator import Orchestrator


@pytest.fixture
def test_settings(tmp_path):
    """Settings pointing at a temporary workspace."""
    return Settings(
        provider=Provider.ANTHROPIC,
        workspace=tmp_path,
        max_cycles=10,
        max_parallel_tasks=2,
    )


# ---------------------------------------------------------------------------
# Pure logic tests (no async, no LLM)
# ---------------------------------------------------------------------------


def test_is_done_empty_tasks():
    """With no tasks stored, _is_done should return True."""
    orch = Orchestrator(goal="test")
    assert orch.tasks == []
    assert orch._is_done() is True


def test_is_done_all_completed():
    """All tasks completed -> done."""
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    assert orch._is_done() is True


def test_is_done_with_failed():
    """Any failed task -> not done (will retry in next cycle)."""
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="failed"),
    ]
    assert orch._is_done() is False


def test_is_done_partial():
    """Partial completion with no failures -> not done."""
    orch = Orchestrator(goal="test")
    orch.tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="pending"),
    ]
    assert orch._is_done() is False


def test_max_parallel_from_settings(test_settings):
    orch = Orchestrator(goal="test", settings=test_settings)
    assert orch._max_parallel() == 2


def test_max_parallel_default():
    """Without settings, falls back to 5."""
    orch = Orchestrator(goal="test")
    assert orch._max_parallel() == 5


def test_get_tasks_returns_stored_tasks():
    """_get_tasks should return the tasks stored from the last plan."""
    orch = Orchestrator(goal="test")
    tasks = [
        TaskModel(id="1", description="a", status="completed"),
        TaskModel(id="2", description="b", status="completed"),
    ]
    orch.tasks = tasks
    assert orch._get_tasks() is tasks


# ---------------------------------------------------------------------------
# Stubs for integration tests
# ---------------------------------------------------------------------------


class StubPlanner:
    """Planner stub that returns *num_tasks* pending tasks per call."""

    def __init__(self, num_tasks=1):
        self.calls = 0
        self.num_tasks = num_tasks

    async def plan(self, goal, max_retries=3):
        self.calls += 1
        tasks = [
            TaskModel(id=str(i + 1), description=f"task {i}", status="pending")
            for i in range(self.num_tasks)
        ]
        return Plan(tasks=tasks, rationale="stub plan")


class StubWorkerAgent:
    """Worker stub that succeeds and writes no files."""

    def __init__(self, task, client=None, settings=None):
        self.task = task

    async def run(self):
        return WorkerResult(summary="stub complete", files=[])


class StubTesterAgent:
    """Tester stub that always passes."""

    def __init__(self, client=None, settings=None):
        pass

    async def run(self, goal, tasks):
        return TestResult(passed=True, summary="all good", failures=[])


@pytest.fixture
def patched_agents():
    """Patch WorkerAgent and TesterAgent in the orchestrator module."""
    original_worker = orch_mod.WorkerAgent
    original_tester = orch_mod.TesterAgent
    orch_mod.WorkerAgent = StubWorkerAgent
    orch_mod.TesterAgent = StubTesterAgent
    try:
        yield
    finally:
        orch_mod.WorkerAgent = original_worker
        orch_mod.TesterAgent = original_tester


def _patch_class(attr_name, new_cls):
    """Set a class attribute on the orchestrator module and return a restorer."""
    original = getattr(orch_mod, attr_name)
    setattr(orch_mod, attr_name, new_cls)

    def _restore():
        setattr(orch_mod, attr_name, original)

    return _restore


# ---------------------------------------------------------------------------
# Integration tests (async)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_stops_when_done(test_settings, patched_agents):
    """When all tasks complete and tests pass, run() should stop after 1 cycle."""
    orch = Orchestrator(goal="quick task", settings=test_settings)
    orch.planner = StubPlanner(num_tasks=2)

    await orch.run()

    assert orch.cycles == 1
    assert orch._is_done() is True
    assert orch.planner.calls == 1


@pytest.mark.asyncio
async def test_run_respects_max_cycles(test_settings):
    """The orchestrator should stop after max_cycles when tasks keep failing."""
    test_settings.max_cycles = 3

    class FailingWorker:
        def __init__(self, task, client=None, settings=None):
            pass

        async def run(self):
            raise RuntimeError("stub failure")

    restore_worker = _patch_class("WorkerAgent", FailingWorker)
    restore_tester = _patch_class("TesterAgent", StubTesterAgent)
    try:
        orch = Orchestrator(goal="keep going", settings=test_settings)
        orch.planner = StubPlanner(num_tasks=1)
        await orch.run()
    finally:
        restore_worker()
        restore_tester()

    assert orch.planner.calls == 3
    assert orch.cycles == 3


@pytest.mark.asyncio
async def test_run_continues_on_test_failure(test_settings):
    """When tests fail, the orchestrator should retry up to max_cycles."""
    test_settings.max_cycles = 2

    class FailingTester:
        def __init__(self, client=None, settings=None):
            pass

        async def run(self, goal, tasks):
            return TestResult(
                passed=False,
                summary="tests failed",
                failures=["test_foo failed"],
            )

    restore_worker = _patch_class("WorkerAgent", StubWorkerAgent)
    restore_tester = _patch_class("TesterAgent", FailingTester)
    try:
        orch = Orchestrator(goal="break things", settings=test_settings)
        orch.planner = StubPlanner(num_tasks=1)
        await orch.run()
    finally:
        restore_worker()
        restore_tester()

    assert orch.planner.calls == 2
    assert orch.cycles == 2
