"""Tests for furrow.core.orchestrator — the infinite loop logic."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import pytest

from furrow.agents.tester import TesterAgent
from furrow.agents.worker import WorkerAgent
from furrow.config import Plan, Settings, TaskModel
from furrow.core.orchestrator import Orchestrator


class _FakeClient:
    """Returns canned JSON responses for planner and tester; plain text for worker.

    The ``complete`` method inspects the prompt to decide which response to
    return:
      * Planner prompt contains "Goal:" (no "Task:" or "Test output:") → plan JSON
      * Worker prompt contains "Task:" → worker summary text
      * Tester prompt contains "Test output:" → test JSON
    """

    def __init__(
        self,
        plan_json: str | None = None,
        test_json: str | None = None,
        worker_response: str = "Worker completed the task.",
        plans: list[str] | None = None,
        test_responses: list[str] | None = None,
    ) -> None:
        self.settings = Settings()
        self._worker_response = worker_response
        # Single-response mode (default).
        self._plan_json = plan_json or json.dumps({
            "tasks": [{"id": "1", "description": "Task one"}],
            "rationale": "Initial plan",
        })
        self._test_json = test_json or json.dumps({
            "passed": True,
            "summary": "All passed",
            "failures": [],
        })
        # Multi-response mode: rotate through lists for sequential calls.
        self._plans = plans or [self._plan_json]
        self._test_responses = test_responses or [self._test_json]
        self._plan_idx = 0
        self._test_idx = 0
        self.complete_calls: list[tuple[str, str, str | None]] = []

    async def complete(self, prompt: str, system: str = "", model: str | None = None) -> str:
        self.complete_calls.append((prompt, system, model))
        if "Test output:" in prompt:
            response = self._test_responses[min(self._test_idx, len(self._test_responses) - 1)]
            self._test_idx += 1
            return response
        if "Task:" in prompt:
            return self._worker_response
        # Otherwise it's a planner prompt.
        response = self._plans[min(self._plan_idx, len(self._plans) - 1)]
        self._plan_idx += 1
        return response


def _stub_run_tests():
    """Patch target: TesterAgent._run_tests returns canned output without running subprocesses."""

    async def _stub(self: TesterAgent) -> str:
        return "stubbed test output"

    return _stub


# ------------------------------------------------------------------ #
# _get_tasks / _is_done
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_get_tasks_empty_before_plan():
    """Before any cycle runs, _get_tasks returns [] and _is_done is True."""
    client = _FakeClient()
    orch = Orchestrator(goal="test goal", client=client)
    assert orch._get_tasks() == []
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_all_completed_tests_pass():
    client = _FakeClient()
    orch = Orchestrator(goal="test", client=client)
    orch._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="x", status="completed"),
            TaskModel(id="2", description="y", status="completed"),
        ],
        rationale="test",
    )
    orch._tests_passed = True
    assert orch._is_done() is True


@pytest.mark.asyncio
async def test_is_done_tests_fail_continues():
    """If tests fail, the loop should continue even with all workers completed."""
    client = _FakeClient()
    orch = Orchestrator(goal="test", client=client)
    orch._current_plan = Plan(
        tasks=[TaskModel(id="1", description="x", status="completed")],
        rationale="test",
    )
    orch._tests_passed = False
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_has_failures():
    """Worker failures should keep the loop going."""
    client = _FakeClient()
    orch = Orchestrator(goal="test", client=client)
    orch._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="x", status="completed"),
            TaskModel(id="2", description="y", status="failed"),
        ],
        rationale="test",
    )
    orch._tests_passed = False
    assert orch._is_done() is False


@pytest.mark.asyncio
async def test_is_done_mixed_incomplete():
    """Pending tasks should keep the loop going."""
    client = _FakeClient()
    orch = Orchestrator(goal="test", client=client)
    orch._current_plan = Plan(
        tasks=[
            TaskModel(id="1", description="x", status="completed"),
            TaskModel(id="2", description="y", status="pending"),
        ],
        rationale="test",
    )
    orch._tests_passed = True
    assert orch._is_done() is False


# ------------------------------------------------------------------ #
# max_cycles enforcement
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_max_cycles_stops_loop():
    """With tests always failing, max_cycles=2 should stop after 2 cycles."""
    plan_json = json.dumps({
        "tasks": [{"id": "1", "description": "do something"}],
        "rationale": "keep going",
    })
    test_json = json.dumps({
        "passed": False,
        "summary": "tests failed",
        "failures": ["something broke"],
    })
    client = _FakeClient(plan_json=plan_json, test_json=test_json)
    orch = Orchestrator(goal="keep building", client=client, max_cycles=2)
    with patch.object(TesterAgent, "_run_tests", new=_stub_run_tests()):
        await orch.run()
    assert orch.cycles == 2
    assert orch.max_cycles == 2


@pytest.mark.asyncio
async def test_max_zero_means_unlimited_then_done():
    """With max_cycles=0 (unlimited), the loop stops when tests pass."""
    test_plan = json.dumps({
        "tasks": [{"id": "1", "description": "do something"}],
        "rationale": "one task",
    })
    test_result = json.dumps({"passed": True, "summary": "All passed", "failures": []})
    client = _FakeClient(plan_json=test_plan, test_json=test_result)
    orch = Orchestrator(goal="finish me", client=client)
    assert orch.max_cycles == 0  # default from settings
    with patch.object(TesterAgent, "_run_tests", new=_stub_run_tests()):
        await orch.run()
    assert orch.cycles == 1


# ------------------------------------------------------------------ #
# max_parallel_tasks enforcement
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_max_parallel_tasks_respected():
    """The semaphore should limit concurrent worker executions."""
    tasks = [{"id": str(i), "description": f"task {i}", "files": [], "dependencies": []} for i in range(3)]
    plan_json = json.dumps({"tasks": tasks, "rationale": "parallel test"})
    test_json = json.dumps({"passed": True, "summary": "ok", "failures": []})
    client = _FakeClient(plan_json=plan_json, test_json=test_json)
    client.settings.max_parallel_tasks = 1

    current = 0
    peak = 0

    async def tracked_run(self_worker: WorkerAgent) -> str:
        nonlocal current, peak
        current += 1
        peak = max(peak, current)
        await asyncio.sleep(0.05)
        current -= 1
        return "done"

    with (
        patch.object(WorkerAgent, "run", tracked_run),
        patch.object(TesterAgent, "_run_tests", new=_stub_run_tests()),
    ):
        orch = Orchestrator(goal="parallel", client=client)
        await orch.run()

    assert orch.cycles == 1
    assert peak <= 1


# ------------------------------------------------------------------ #
# Full cycle integration
# ------------------------------------------------------------------ #

@pytest.mark.asyncio
async def test_full_cycle_all_pass():
    """A successful cycle: plan → execute → test pass → stop."""
    plan_json = json.dumps({
        "tasks": [{"id": "1", "description": "Implement feature X"}],
        "rationale": "one task",
    })
    test_json = json.dumps({"passed": True, "summary": "All tests green", "failures": []})
    client = _FakeClient(plan_json=plan_json, test_json=test_json)
    orch = Orchestrator(goal="Build feature X", client=client)
    with patch.object(TesterAgent, "_run_tests", new=_stub_run_tests()):
        results = await orch.run()
    assert len(results) == 1
    assert results[0].status == "completed"
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_failed_workers_continue_next_cycle():
    """When all workers fail, the loop continues; next cycle tasks succeed and tests pass."""
    plan_json = json.dumps({
        "tasks": [{"id": "1", "description": "Fix bug"}],
        "rationale": "fix attempt",
    })
    test_pass = json.dumps({"passed": True, "summary": "ok", "failures": []})
    client = _FakeClient(plan_json=plan_json, test_json=test_pass, worker_response=None)

    call_count = {"worker": 0}

    async def failing_then_successful(self_worker: WorkerAgent) -> str:
        call_count["worker"] += 1
        if call_count["worker"] == 1:
            raise RuntimeError("Worker crashed on first attempt")
        return "done"

    with (
        patch.object(WorkerAgent, "run", failing_then_successful),
        patch.object(TesterAgent, "_run_tests", new=_stub_run_tests()),
    ):
        orch = Orchestrator(goal="Fix the bug", client=client, max_cycles=5)
        results = await orch.run()

    # Cycle 1: worker fails (status="failed"), tests still run and pass.
    #   _is_done → failed>0 → False → continue.
    # Cycle 2: worker succeeds (status="completed"), tests pass.
    #   _is_done → all completed + tests_passed → True → stop.
    assert orch.cycles == 2
    # Final cycle task should be completed.
    assert all(t.status == "completed" for t in results)


@pytest.mark.asyncio
async def test_no_tasks_planned_stops():
    """When the planner returns zero tasks, the loop should stop."""
    plan_json = json.dumps({"tasks": [], "rationale": "nothing to do"})
    client = _FakeClient(plan_json=plan_json)
    orch = Orchestrator(goal="already done", client=client)
    with patch.object(TesterAgent, "_run_tests", new=_stub_run_tests()):
        results = await orch.run()
    assert len(results) == 0
    assert orch.cycles == 1


@pytest.mark.asyncio
async def test_get_tasks_returns_plan_tasks():
    """After a cycle, _get_tasks should return the plan's tasks."""
    plan_json = json.dumps({
        "tasks": [{"id": "1", "description": "Task one"}, {"id": "2", "description": "Task two"}],
        "rationale": "two tasks",
    })
    test_json = json.dumps({"passed": True, "summary": "ok", "failures": []})
    client = _FakeClient(plan_json=plan_json, test_json=test_json)
    orch = Orchestrator(goal="build", client=client)
    with patch.object(TesterAgent, "_run_tests", new=_stub_run_tests()):
        await orch.run()
    tasks = orch._get_tasks()
    assert len(tasks) == 2
    assert tasks[0].description == "Task one"
    assert tasks[1].description == "Task two"
