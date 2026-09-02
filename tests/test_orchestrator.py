"""Tests for the Orchestrator."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from furrow.config import Plan, Settings, TaskModel, TestResult
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import SessionStatus


# ---------------------------------------------------------------------------
# Mock agents for testing the orchestrator without real LLM calls
# ---------------------------------------------------------------------------


class MockPlannerAgent:
    """Planner that returns a pre-built plan."""

    def __init__(self, plan: Plan):
        self._plan = plan
        self.client = None  # type: ignore

    async def plan(self, goal: str) -> Plan:
        return self._plan


@pytest.fixture
def tmp_state_file(tmp_path) -> str:
    return str(tmp_path / ".furrow" / "state.json")


@pytest.fixture
def orchestrator(tmp_state_file) -> Orchestrator:
    return Orchestrator(goal="Test goal", state_file=tmp_state_file)


# ---------------------------------------------------------------------------
# Init tests
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    def test_init_creates_state(self, orchestrator: Orchestrator) -> None:
        assert orchestrator.state.goal == "Test goal"
        assert orchestrator.state.status == SessionStatus.ACTIVE
        assert orchestrator.state.cycle == 0

    def test_init_sets_max_cycles(self, tmp_state_file) -> None:
        settings = Settings(max_cycles=5)
        orch = Orchestrator(goal="Test", state_file=tmp_state_file, settings=settings)
        assert orch.state.max_cycles == 5

    def test_original_goal_preserved(self, orchestrator: Orchestrator) -> None:
        assert orchestrator.original_goal == "Test goal"
        assert orchestrator.state.original_goal == "Test goal"

    def test_init_loads_existing_state(self, tmp_state_file) -> None:
        """Orch should load existing state if present."""
        orch1 = Orchestrator(goal="First goal", state_file=tmp_state_file)
        orch1.state_manager.increment_cycle()
        orch1.state_manager.save()

        orch2 = Orchestrator(goal="Second goal", state_file=tmp_state_file)
        assert orch2.state.cycle == 1

    def test_state_file_created_on_init(self, orchestrator: Orchestrator, tmp_state_file) -> None:
        assert Path(tmp_state_file).exists()


# ---------------------------------------------------------------------------
# _is_done tests
# ---------------------------------------------------------------------------


class TestOrchestratorIsDone:
    def test_is_done_no_tasks(self, orchestrator: Orchestrator) -> None:
        # No tasks means goal was complete (planner returned empty)
        orchestrator.state.status = SessionStatus.COMPLETED
        assert orchestrator._is_done() is True

    def test_is_done_pending_tasks(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.tasks = [
            TaskModel(id="1", description="Task", status="pending"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_no_tests_yet(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.tasks = [
            TaskModel(id="1", description="Task", status="completed"),
        ]
        assert orchestrator._is_done() is False

    def test_is_done_all_complete_tests_pass(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.tasks = [
            TaskModel(id="1", description="Task", status="completed"),
        ]
        orchestrator.state.test_history = [
            {"cycle": 1, "passed": True, "summary": "ok", "failures": []}
        ]
        assert orchestrator._is_done() is True

    def test_is_done_all_complete_tests_fail(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.tasks = [
            TaskModel(id="1", description="Task", status="completed"),
        ]
        orchestrator.state.test_history = [
            {"cycle": 1, "passed": False, "summary": "fail", "failures": ["x"]}
        ]
        assert orchestrator._is_done() is False

    def test_is_done_failed_tasks(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.tasks = [
            TaskModel(id="1", description="Task", status="failed"),
        ]
        orchestrator.state.test_history = [
            {"cycle": 1, "passed": True, "summary": "ok", "failures": []}
        ]
        assert orchestrator._is_done() is False

    def test_is_done_mixed_pending_and_completed(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.tasks = [
            TaskModel(id="1", description="Task A", status="completed"),
            TaskModel(id="2", description="Task B", status="pending"),
        ]
        orchestrator.state.test_history = [
            {"cycle": 1, "passed": True, "summary": "ok", "failures": []}
        ]
        assert orchestrator._is_done() is False


# ---------------------------------------------------------------------------
# Cycle limit tests
# ---------------------------------------------------------------------------


class TestOrchestratorCycleLimit:
    def test_cycle_limit_reached(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.max_cycles = 3
        for _ in range(3):
            orchestrator.state_manager.increment_cycle()
        assert orchestrator.state_manager.is_cycle_limit_reached() is True

    def test_cycle_limit_not_reached(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.max_cycles = 3
        orchestrator.state_manager.increment_cycle()
        assert orchestrator.state_manager.is_cycle_limit_reached() is False

    def test_no_limit_when_max_zero(self, orchestrator: Orchestrator) -> None:
        orchestrator.state.max_cycles = 0
        orchestrator.state_manager.increment_cycle()
        assert orchestrator.state_manager.is_cycle_limit_reached() is False

    def test_run_halts_at_cycle_limit(self, tmp_state_file) -> None:
        """Full run should stop when max_cycles is reached (with failing tests)."""
        plan = Plan(
            tasks=[TaskModel(id="1", description="Task")],
            rationale="Test plan",
        )
        orch = Orchestrator(goal="Stop after 1 cycle", state_file=tmp_state_file)
        orch.planner = MockPlannerAgent(plan)
        orch.state.max_cycles = 1

        with patch(
            "furrow.core.orchestrator.WorkerAgent"
        ) as MockWorker, patch(
            "furrow.core.orchestrator.TesterAgent"
        ) as MockTester:
            MockWorker.return_value.run = AsyncMock(return_value="Completed")
            # Use failing tests so the loop continues to the next iteration
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=False, summary="broken", failures=["err"])
            )
            state = asyncio.run(orch.run())

        assert state.status == SessionStatus.FAILED
        assert any("Max cycles" in e for e in state.errors)


# ---------------------------------------------------------------------------
# Full cycle tests
# ---------------------------------------------------------------------------


class TestOrchestratorRun:
    def test_full_cycle_all_pass(self, tmp_state_file) -> None:
        """End-to-end test: plan → execute → test all pass → done."""
        plan = Plan(
            tasks=[
                TaskModel(id="1", description="Task A"),
                TaskModel(id="2", description="Task B"),
            ],
            rationale="Complete plan",
        )

        orch = Orchestrator(goal="Complete goal", state_file=tmp_state_file)
        orch.planner = MockPlannerAgent(plan)

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockWorker.return_value.run = AsyncMock(return_value="Completed")
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=True, summary="All passed", failures=[])
            )
            state = asyncio.run(orch.run())

        assert state.status == SessionStatus.COMPLETED
        assert orch.state.cycle >= 1
        assert orch.state_manager.completed_count() >= 1

    def test_cycle_continues_on_test_failure(self, tmp_state_file) -> None:
        """When tests fail, the loop should retry (not stop)."""
        plan = Plan(
            tasks=[TaskModel(id="1", description="Fix tests")],
            rationale="Fix plan",
        )

        orch = Orchestrator(goal="Fix goal", state_file=tmp_state_file)
        orch.planner = MockPlannerAgent(plan)
        orch.state.max_cycles = 1  # Stop after 1 cycle to avoid infinite loop

        with patch("furrow.core.orchestrator.WorkerAgent") as MockWorker, \
             patch("furrow.core.orchestrator.TesterAgent") as MockTester:
            MockWorker.return_value.run = AsyncMock(return_value="Completed")
            MockTester.return_value.run = AsyncMock(
                return_value=TestResult(passed=False, summary="Tests broken", failures=["test_x failed"])
            )
            state = asyncio.run(orch.run())

        # Should have stopped due to cycle limit, with failed status
        assert state.status == SessionStatus.FAILED
        # Goal should have been updated to focus on fixing
        assert "Fix failing tests" in orch.state.goal

    def test_state_persisted_after_run(self, tmp_state_file) -> None:
        """State should be saved to disk after run."""
        plan = Plan(tasks=[], rationale="Done")
        orch = Orchestrator(goal="Empty goal", state_file=tmp_state_file)
        orch.planner = MockPlannerAgent(plan)

        state = asyncio.run(orch.run())
        assert state.status == SessionStatus.COMPLETED

        # Verify state file exists
        assert Path(tmp_state_file).exists()

    def test_state_property_delegates(self, orchestrator: Orchestrator) -> None:
        assert orchestrator.state is orchestrator.state_manager.state

    def test_run_with_no_tasks_completes(self, tmp_state_file) -> None:
        """When planner returns no tasks, run should complete immediately."""
        orch = Orchestrator(goal="Already done", state_file=tmp_state_file)
        orch.planner = MockPlannerAgent(Plan(tasks=[], rationale="Goal is complete"))

        state = asyncio.run(orch.run())
        assert state.status == SessionStatus.COMPLETED
        assert orch.state.cycle == 1
