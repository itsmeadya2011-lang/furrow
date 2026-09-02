"""Tests for the StateManager and SessionState persistence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from furrow.config import TaskModel
from furrow.core.state import SessionState, SessionStatus, StateManager


@pytest.fixture
def temp_state_file(tmp_path: Path) -> Path:
    """Provide a temporary state file path."""
    return tmp_path / ".furrow" / "state.json"


@pytest.fixture
def sm(temp_state_file: Path) -> StateManager:
    return StateManager(state_file=temp_state_file)


@pytest.fixture
def populated_sm(sm: StateManager) -> StateManager:
    """A StateManager initialized with a sample goal."""
    sm.initialize(goal="Add unit tests to the codebase", max_cycles=5)
    return sm


class TestSessionState:
    def test_to_dict_roundtrip(self, populated_sm: StateManager) -> None:
        state = populated_sm.state
        data = state.to_dict()
        assert data["goal"] == "Add unit tests to the codebase"
        assert data["status"] == SessionStatus.ACTIVE.value
        assert data["cycle"] == 0
        assert data["max_cycles"] == 5
        assert data["tasks"] == []

    def test_from_dict_reconstructs(self, populated_sm: StateManager) -> None:
        state = populated_sm.state
        data = state.to_dict()
        restored = SessionState.from_dict(data)
        assert restored.goal == state.goal
        assert restored.status == state.status
        assert restored.cycle == state.cycle
        assert restored.max_cycles == state.max_cycles


class TestStateManagerInit:
    def test_initialize_creates_state_file(self, sm: StateManager, temp_state_file: Path) -> None:
        state = sm.initialize(goal="Test goal", max_cycles=10)
        assert state.goal == "Test goal"
        assert state.max_cycles == 10
        assert temp_state_file.exists()

    def test_initialize_sets_active_status(self, sm: StateManager) -> None:
        state = sm.initialize(goal="Test", max_cycles=0)
        assert state.status == SessionStatus.ACTIVE

    def test_load_returns_none_when_no_file(self, sm: StateManager) -> None:
        assert sm.load() is None

    def test_load_returns_none_for_corrupt_file(self, sm: StateManager, temp_state_file: Path) -> None:
        temp_state_file.parent.mkdir(parents=True, exist_ok=True)
        temp_state_file.write_text("{ invalid json }")
        assert sm.load() is None

    def test_state_property_raises_before_init(self, sm: StateManager) -> None:
        """Accessing state before initialize/load should raise RuntimeError."""
        with pytest.raises(RuntimeError, match="not been initialized"):
            _ = sm.state

    def test_state_property_lazy_loads(self, populated_sm: StateManager) -> None:
        # After initialize, state should be available
        assert populated_sm.state.goal == "Add unit tests to the codebase"

    def test_save_persists_to_disk(self, populated_sm: StateManager, temp_state_file: Path) -> None:
        populated_sm.state.cycle = 3
        populated_sm.save()
        data = json.loads(temp_state_file.read_text())
        assert data["cycle"] == 3


class TestStateManagerTasks:
    def test_update_tasks_replaces_list(self, populated_sm: StateManager) -> None:
        tasks = [
            TaskModel(id="1", description="Task A", status="completed"),
            TaskModel(id="2", description="Task B", status="pending"),
        ]
        populated_sm.update_tasks(tasks)
        assert len(populated_sm.state.tasks) == 2
        assert populated_sm.state.tasks[0].description == "Task A"

    def test_update_tasks_preserves_completed_status(self, populated_sm: StateManager) -> None:
        completed_task = TaskModel(id="1", description="Task A", status="completed", result="done")
        populated_sm.update_tasks([completed_task])

        # New plan with same task id but pending status
        new_plan = [TaskModel(id="1", description="Task A", status="pending")]
        populated_sm.update_tasks(new_plan)
        # Completed status should be preserved
        assert populated_sm.state.tasks[0].status == "completed"
        assert populated_sm.state.tasks[0].result == "done"

    def test_mark_task_completed(self, populated_sm: StateManager) -> None:
        task = TaskModel(id="1", description="Do something")
        populated_sm.update_tasks([task])
        populated_sm.mark_task_completed("1", "Changed file X")
        assert populated_sm.state.tasks[0].status == "completed"
        assert populated_sm.state.tasks[0].result == "Changed file X"

    def test_mark_task_completed_nonexistent_id(self, populated_sm: StateManager) -> None:
        """Should not raise when marking a non-existent task."""
        populated_sm.mark_task_completed("999", "result")
        # No crash

    def test_mark_task_failed(self, populated_sm: StateManager) -> None:
        task = TaskModel(id="1", description="Do something")
        populated_sm.update_tasks([task])
        populated_sm.mark_task_failed("1", "Error occurred")
        assert populated_sm.state.tasks[0].status == "failed"
        assert populated_sm.state.tasks[0].result == "Error occurred"

    def test_all_tasks_done_all_completed(self, populated_sm: StateManager) -> None:
        tasks = [
            TaskModel(id="1", description="Task A", status="completed"),
            TaskModel(id="2", description="Task B", status="completed"),
        ]
        populated_sm.update_tasks(tasks)
        assert populated_sm.all_tasks_done() is True

    def test_all_tasks_done_with_pending(self, populated_sm: StateManager) -> None:
        tasks = [
            TaskModel(id="1", description="Task A", status="completed"),
            TaskModel(id="2", description="Task B", status="pending"),
        ]
        populated_sm.update_tasks(tasks)
        assert populated_sm.all_tasks_done() is False

    def test_all_tasks_done_empty(self, populated_sm: StateManager) -> None:
        assert populated_sm.all_tasks_done() is False

    def test_has_failures(self, populated_sm: StateManager) -> None:
        tasks = [TaskModel(id="1", description="Task A", status="failed")]
        populated_sm.update_tasks(tasks)
        assert populated_sm.has_failures() is True

    def test_no_failures(self, populated_sm: StateManager) -> None:
        tasks = [TaskModel(id="1", description="Task A", status="completed")]
        populated_sm.update_tasks(tasks)
        assert populated_sm.has_failures() is False

    def test_completed_count(self, populated_sm: StateManager) -> None:
        tasks = [
            TaskModel(id="1", description="A", status="completed"),
            TaskModel(id="2", description="B", status="failed"),
            TaskModel(id="3", description="C", status="pending"),
        ]
        populated_sm.update_tasks(tasks)
        assert populated_sm.completed_count() == 1
        assert populated_sm.failed_count() == 1

    def test_get_task_found(self, populated_sm: StateManager) -> None:
        task = TaskModel(id="abc", description="Find me")
        populated_sm.update_tasks([task])
        found = populated_sm.get_task("abc")
        assert found is not None
        assert found.description == "Find me"

    def test_get_task_not_found(self, populated_sm: StateManager) -> None:
        assert populated_sm.get_task("nonexistent") is None


class TestStateManagerCycle:
    def test_increment_cycle(self, populated_sm: StateManager) -> None:
        assert populated_sm.state.cycle == 0
        populated_sm.increment_cycle()
        assert populated_sm.state.cycle == 1
        populated_sm.increment_cycle()
        assert populated_sm.state.cycle == 2

    def test_is_cycle_limit_reached_no_limit(self, populated_sm: StateManager) -> None:
        populated_sm.state.max_cycles = 0
        assert populated_sm.is_cycle_limit_reached() is False

    def test_is_cycle_limit_reached_within_limit(self, populated_sm: StateManager) -> None:
        populated_sm.state.max_cycles = 5
        for _ in range(4):
            populated_sm.increment_cycle()
        assert populated_sm.is_cycle_limit_reached() is False

    def test_is_cycle_limit_reached_at_limit(self, populated_sm: StateManager) -> None:
        populated_sm.state.max_cycles = 3
        for _ in range(3):
            populated_sm.increment_cycle()
        assert populated_sm.is_cycle_limit_reached() is True


class TestStateManagerErrors:
    def test_add_error(self, populated_sm: StateManager) -> None:
        populated_sm.add_error("Something went wrong")
        assert len(populated_sm.state.errors) == 1
        assert "Something went wrong" in populated_sm.state.errors[0]

    def test_set_goal(self, populated_sm: StateManager) -> None:
        populated_sm.set_goal("New goal after failure")
        assert populated_sm.state.goal == "New goal after failure"

    def test_complete_sets_status(self, populated_sm: StateManager) -> None:
        populated_sm.complete()
        assert populated_sm.state.status == SessionStatus.COMPLETED

    def test_fail_sets_status_and_error(self, populated_sm: StateManager) -> None:
        populated_sm.fail("Test failure reason")
        assert populated_sm.state.status == SessionStatus.FAILED
        assert "Test failure reason" in populated_sm.state.errors


class TestStateManagerPersistence:
    def test_persistence_roundtrip(self, temp_state_file: Path) -> None:
        """Ensure state survives save → load roundtrip."""
        sm1 = StateManager(state_file=temp_state_file)
        sm1.initialize(goal="Persistent goal", max_cycles=2)
        # Add tasks before marking completed
        sm1.update_tasks([TaskModel(id="1", description="Task A", status="pending")])
        sm1.increment_cycle()
        sm1.mark_task_completed("1", "Done")
        sm1.save()

        # Create a new manager pointing to the same file
        sm2 = StateManager(state_file=temp_state_file)
        loaded = sm2.load()
        assert loaded is not None
        assert loaded.goal == "Persistent goal"
        assert loaded.cycle == 1
        assert loaded.tasks[0].status == "completed"
        assert loaded.tasks[0].result == "Done"

    def test_resume_preserves_all_state(self, temp_state_file: Path) -> None:
        sm = StateManager(state_file=temp_state_file)
        sm.initialize(goal="Resume test", max_cycles=10)
        sm.increment_cycle()
        sm.add_error("Historical error")
        sm.save()

        sm2 = StateManager(state_file=temp_state_file)
        state = sm2.load()
        assert state is not None
        assert state.goal == "Resume test"
        assert state.max_cycles == 10
        assert state.cycle == 1
        assert state.errors == ["Historical error"]
