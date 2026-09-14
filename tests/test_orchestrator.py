import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from furrow.core.orchestrator import Orchestrator
from furrow.config import TaskModel


class MockLLMClient:
    """Simple mock LLMClient."""

    def __init__(self, *args, **kwargs) -> None:
        pass


class MockPlannerAgent:
    """Simple mock PlannerAgent."""

    def __init__(self, *args, **kwargs) -> None:
        pass


class TestOrchestratorTasks:
    """Tests for Orchestrator task tracking."""

    def test_get_tasks_returns_accumulated(self) -> None:
        """_get_tasks returns tasks accumulated across cycles."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            orch = Orchestrator(goal="test goal")
            orch.tasks = [
                TaskModel(id="1", description="task 1"),
                TaskModel(id="2", description="task 2"),
            ]
            assert orch._get_tasks() == orch.tasks
            assert len(orch._get_tasks()) == 2


class TestOrchestratorIsDone:
    """Tests for Orchestrator._is_done."""

    def test_is_done_false_when_no_tasks(self) -> None:
        """_is_done returns False when no tasks exist."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            orch = Orchestrator(goal="test goal")
            assert orch._is_done() is False

    def test_is_done_true_when_all_completed(self) -> None:
        """_is_done returns True when all tasks are completed."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            orch = Orchestrator(goal="test goal")
            orch.tasks = [
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="completed"),
            ]
            assert orch._is_done() is True

    def test_is_done_false_when_any_failed(self) -> None:
        """_is_done returns False when any task has failed."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            orch = Orchestrator(goal="test goal")
            orch.tasks = [
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="failed"),
            ]
            assert orch._is_done() is False


class TestOrchestratorReset:
    """Tests for Orchestrator.reset."""

    def test_reset_clears_tasks_and_cycles(self) -> None:
        """reset clears tasks and cycle counter."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            orch = Orchestrator(goal="original goal")
            orch.tasks = [TaskModel(id="1", description="task 1", status="completed")]
            orch.cycles = 5
            orch.reset()
            assert orch.tasks == []
            assert orch.cycles == 0
            assert orch.goal == "original goal"

    def test_reset_with_new_goal(self) -> None:
        """reset with a new goal replaces the goal."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            orch = Orchestrator(goal="original goal")
            orch.reset(goal="new goal")
            assert orch.goal == "new goal"


class TestOrchestratorState:
    """Tests for Orchestrator state persistence."""

    async def test_save_and_load_state_roundtrip(self, tmp_path: Path) -> None:
        """save_state and load_state round-trip preserves data."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            state_file = tmp_path / "state.json"
            orch = Orchestrator(goal="test goal", state_file=str(state_file))
            orch.tasks = [
                TaskModel(id="1", description="task 1", status="completed"),
                TaskModel(id="2", description="task 2", status="pending"),
            ]
            orch.cycles = 3

            await orch.save_state()
            assert state_file.exists()

            # Create new orchestrator and load state
            orch2 = Orchestrator(goal="other", state_file=str(state_file))
            await orch2.load_state()
            assert orch2.goal == "test goal"
            assert orch2.cycles == 3
            assert len(orch2.tasks) == 2
            assert orch2.tasks[0].id == "1"
            assert orch2.tasks[0].status == "completed"
            assert orch2.tasks[1].id == "2"
            assert orch2.tasks[1].status == "pending"

    async def test_load_state_no_file(self, tmp_path: Path) -> None:
        """load_state does nothing when state file doesn't exist."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            state_file = tmp_path / "nonexistent.json"
            orch = Orchestrator(goal="test goal", state_file=str(state_file))
            await orch.load_state()
            assert orch.goal == "test goal"
            assert orch.cycles == 0
            assert orch.tasks == []

    def test_clear_state_deletes_file(self, tmp_path: Path) -> None:
        """clear_state deletes the state file."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            state_file = tmp_path / "state.json"
            state_file.write_text("{}", encoding="utf-8")
            orch = Orchestrator(goal="test goal", state_file=str(state_file))
            assert state_file.exists()
            orch.clear_state()
            assert not state_file.exists()

    def test_clear_state_no_file(self, tmp_path: Path) -> None:
        """clear_state does nothing when state file doesn't exist."""
        with patch("furrow.core.orchestrator.LLMClient", MockLLMClient), patch(
            "furrow.core.orchestrator.PlannerAgent", MockPlannerAgent
        ):
            state_file = tmp_path / "nonexistent.json"
            orch = Orchestrator(goal="test goal", state_file=str(state_file))
            orch.clear_state()
            assert not state_file.exists()