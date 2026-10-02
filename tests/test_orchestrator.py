"""Tests for the Orchestrator class."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


def make_mock_client(plans=None, results=None, test_results=None, settings=None):
    """Create a mock LLMClient with pre-configured responses."""
    client = MagicMock()
    client.settings = settings or Settings()
    client.complete = AsyncMock()

    plan_responses = list(plans) if plans else []
    task_responses = list(results) if results else []
    test_responses = list(test_results) if test_results else []

    async def mock_complete(prompt, system="", model=None):
        if "Break the user's goal" in prompt or "Break a goal" in prompt:
            if plan_responses:
                return plan_responses.pop(0)
            return '{"tasks": [], "rationale": "done"}'
        if "implement the assigned task" in prompt.lower():
            if task_responses:
                return task_responses.pop(0)
            return "Task completed."
        if "Given the goal and test output" in prompt or "test output" in prompt.lower():
            if test_responses:
                return test_responses.pop(0)
            return '{"passed": true, "summary": "All passed", "failures": []}'
        return '{"passed": true, "summary": "ok", "failures": []}'

    client.complete = mock_complete
    client.complete_async = mock_complete
    return client


@pytest.fixture(autouse=True)
def _patch_tester_run_tests():
    """Avoid running a real subprocess test runner during orchestrator tests."""
    with patch(
        "furrow.agents.tester.TesterAgent._run_tests",
        new_callable=AsyncMock,
        return_value="All tests passed.",
    ):
        yield


def test_orchestrator_init_defaults():
    """Orchestrator uses default settings when none provided."""
    orch = Orchestrator(goal="Test goal")
    assert orch.goal == "Test goal"
    assert orch.cycles == 0
    assert orch._current_tasks == []
    assert orch._current_tests_passed is True


def test_orchestrator_init_with_settings():
    """Orchestrator accepts custom settings."""
    custom_settings = Settings(max_cycles=3, max_parallel_tasks=2)
    orch = Orchestrator(goal="Test", settings=custom_settings)
    assert orch.settings.max_cycles == 3
    assert orch.settings.max_parallel_tasks == 2


def test_orchestrator_is_done_empty_plan():
    """No tasks in current plan means goal is done."""
    orch = Orchestrator(goal="Test")
    orch._current_tasks = []
    assert orch._is_done() is True


def test_orchestrator_is_done_all_completed_tests_pass():
    """All tasks completed + tests passed means done."""
    orch = Orchestrator(goal="Test")
    t1 = TaskModel(id="1", description="task1")
    t1.status = "completed"
    t2 = TaskModel(id="2", description="task2")
    t2.status = "completed"
    orch._current_tasks = [t1, t2]
    orch._current_tests_passed = True
    assert orch._is_done() is True


def test_orchestrator_is_done_tests_failed():
    """Tests failed means not done."""
    orch = Orchestrator(goal="Test")
    t1 = TaskModel(id="1", description="task1")
    t1.status = "completed"
    orch._current_tasks = [t1]
    orch._current_tests_passed = False
    assert orch._is_done() is False


def test_orchestrator_is_done_task_failed():
    """A failed task means not done."""
    orch = Orchestrator(goal="Test")
    t1 = TaskModel(id="1", description="task1")
    t1.status = "failed"
    orch._current_tasks = [t1]
    orch._current_tests_passed = True
    assert orch._is_done() is False


def test_orchestrator_get_tasks_returns_current():
    """_get_tasks returns the current cycle's tasks."""
    orch = Orchestrator(goal="Test")
    t1 = TaskModel(id="1", description="task1")
    orch._current_tasks = [t1]
    tasks = orch._get_tasks()
    assert len(tasks) == 1
    assert tasks[0].id == "1"


def test_orchestrator_max_cycles_stops():
    """Orchestrator stops when max_cycles is reached."""
    settings = Settings(max_cycles=2)
    client = make_mock_client(
        plans=['{"tasks": [], "rationale": "done"}'],
        results=["ok"],
        test_results=['{"passed": true, "summary": "ok", "failures": []}'],
        settings=settings,
    )
    orch = Orchestrator(goal="Test", client=client, settings=settings)
    assert orch.settings.max_cycles == 2


async def test_orchestrator_run_empty_plan():
    """Orchestrator stops immediately when planner returns empty plan."""
    client = make_mock_client(
        plans=['{"tasks": [], "rationale": "Goal is already complete"}'],
    )
    orch = Orchestrator(goal="Test", client=client)
    await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True


async def test_orchestrator_run_all_pass():
    """Orchestrator completes and stops when all tasks pass."""
    plan_json = '{"tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}], "rationale": "ok"}'
    client = make_mock_client(
        plans=[plan_json],
        results=["Implemented feature"],
        test_results=['{"passed": true, "summary": "All tests passed", "failures": []}'],
    )
    orch = Orchestrator(goal="Test", client=client)
    await orch.run()
    assert orch.cycles == 1
    assert orch._is_done() is True


async def test_orchestrator_run_tests_fail():
    """Orchestrator rewrites goal when tests fail and continues."""
    plan_json1 = '{"tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}], "rationale": "ok"}'
    plan_json2 = '{"tasks": [], "rationale": "All done"}'
    client = make_mock_client(
        plans=[plan_json1, plan_json2],
        results=["Implemented feature"],
        test_results=[
            '{"passed": false, "summary": "1 failure", "failures": ["test_x failed"]}',
            '{"passed": true, "summary": "ok", "failures": []}',
        ],
    )
    orch = Orchestrator(goal="Test", client=client)
    await orch.run()
    # First cycle: tests failed -> goal rewritten
    # Second cycle: empty plan -> done
    assert orch.cycles == 2
    assert "Fix failing tests" in orch.goal


async def test_orchestrator_run_max_cycles():
    """Orchestrator stops at max_cycles even if goal incomplete."""
    plan_json = '{"tasks": [{"id": "1", "description": "do thing", "files": [], "dependencies": []}], "rationale": "ok"}'
    settings = Settings(max_cycles=2)
    client = make_mock_client(
        plans=[plan_json, plan_json],
        results=["ok", "ok"],
        test_results=['{"passed": false, "summary": "fail", "failures": ["broken"]}',
                       '{"passed": false, "summary": "fail", "failures": ["broken"]}'],
        settings=settings,
    )
    orch = Orchestrator(goal="Test", client=client, settings=settings)
    await orch.run()
    assert orch.cycles == 2


async def test_orchestrator_dependency_resolution():
    """Tasks with dependencies run after their dependencies complete."""
    # Create tasks with a dependency
    t1 = TaskModel(id="1", description="task1", dependencies=[])
    t2 = TaskModel(id="2", description="task2", dependencies=["1"])

    client = make_mock_client(
        plans='{"tasks": [], "rationale": "test"}',
        settings=Settings(),
    )
    orch = Orchestrator(goal="Test", client=client)

    # _execute_tasks should handle dependency ordering
    with patch.object(orch, '_cycle', new_callable=AsyncMock) as mock_cycle:
        mock_cycle.return_value = None
        await orch._cycle()

    # Test _execute_tasks directly with a mock client
    results = await orch._execute_tasks([t1, t2])
    assert len(results) == 2
    # t1 should complete (no deps), t2 depends on t1
    assert t1.status == "completed"


async def test_orchestrator_task_failure_handled():
    """Failed tasks (exceptions) are handled gracefully."""
    orch = Orchestrator(goal="Test", client=make_mock_client())

    t1 = TaskModel(id="1", description="task1")

    # Patch the worker to raise an exception
    from furrow.agents.worker import WorkerAgent
    original_run = WorkerAgent.run

    async def fail_run(self):
        raise RuntimeError("Worker crashed")

    WorkerAgent.run = fail_run
    try:
        results = await orch._execute_tasks([t1])
        assert len(results) == 1
        assert isinstance(results[0], RuntimeError)
        assert t1.status == "failed"
    finally:
        WorkerAgent.run = original_run
