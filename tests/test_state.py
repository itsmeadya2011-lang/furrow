from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from furrow.config import Plan, TaskModel
from furrow.core.state import RunState, StateManager, TaskState


class TestRunState:
    """Tests for RunState."""

    def test_run_state_new_creates_timestamps(self) -> None:
        """RunState.new() creates valid ISO timestamps."""
        state = RunState.new(goal="Test goal")

        assert state.goal == "Test goal"
        assert state.cycles == 0
        assert state.tasks == []
        assert state.history == []

        # Verify timestamps are valid ISO format
        created = datetime.fromisoformat(state.created_at)
        updated = datetime.fromisoformat(state.updated_at)
        assert isinstance(created, datetime)
        assert isinstance(updated, datetime)

    def test_run_state_new_with_tasks(self, sample_plan: Plan) -> None:
        """RunState.new() accepts tasks parameter."""
        tasks = [
            TaskState(
                id=task.id,
                description=task.description,
                files=task.files,
                status=task.status,
                result=task.result,
            )
            for task in sample_plan.tasks
        ]
        state = RunState.new(goal="Test goal", tasks=tasks)

        assert len(state.tasks) == 2
        assert state.tasks[0].id == "task-1"
        assert state.tasks[1].id == "task-2"


class TestStateManager:
    """Tests for StateManager."""

    def test_default_path(self) -> None:
        """Default path is .furrow-state.json."""
        manager = StateManager()
        assert manager.path.name == ".furrow-state.json"

    def test_string_path(self, tmp_path: Path) -> None:
        """String path works correctly."""
        path_str = str(tmp_path / "custom-state.json")
        manager = StateManager(path=path_str)
        assert manager.path == Path(path_str)

    def test_state_manager_load_missing_file(self, tmp_path: Path) -> None:
        """Load returns None when file doesn't exist."""
        manager = StateManager(path=tmp_path / "nonexistent.json")
        result = manager.load()
        assert result is None

    def test_state_manager_load_corrupt_json(self, tmp_path: Path) -> None:
        """Load returns None for corrupt JSON."""
        corrupt_file = tmp_path / "corrupt.json"
        corrupt_file.write_text("{ invalid json }", encoding="utf-8")

        manager = StateManager(path=corrupt_file)
        result = manager.load()
        assert result is None

    def test_state_manager_save_creates_file(self, state_manager: StateManager, sample_run_state: RunState) -> None:
        """save() creates the file."""
        assert not state_manager.path.exists()
        state_manager.save(sample_run_state)
        assert state_manager.path.exists()

    def test_state_manager_save_load_roundtrip(
        self, state_manager: StateManager, sample_run_state: RunState
    ) -> None:
        """Save state, load it back, verify fields match."""
        state_manager.save(sample_run_state)
        loaded = state_manager.load()

        assert loaded is not None
        assert loaded.goal == sample_run_state.goal
        assert loaded.cycles == sample_run_state.cycles
        assert len(loaded.tasks) == len(sample_run_state.tasks)
        assert loaded.history == sample_run_state.history
        # Timestamps will differ (updated_at is set on save)
        assert loaded.created_at == sample_run_state.created_at

    def test_state_manager_save_updates_timestamp(
        self, state_manager: StateManager, sample_run_state: RunState
    ) -> None:
        """updated_at changes on re-save."""
        state_manager.save(sample_run_state)
        first_save = state_manager.load()
        assert first_save is not None
        first_updated = first_save.updated_at

        # Modify and save again
        first_save.cycles = 5
        state_manager.save(first_save)
        second_save = state_manager.load()
        assert second_save is not None

        assert second_save.updated_at != first_updated
        assert second_save.cycles == 5

    def test_from_plan(self, sample_plan: Plan) -> None:
        """from_plan creates correct RunState from a Plan with tasks."""
        state = StateManager.from_plan(sample_plan, "Test goal")

        assert state.goal == "Test goal"
        assert len(state.tasks) == 2
        assert state.tasks[0].id == "task-1"
        assert state.tasks[0].description == "First task"
        assert state.tasks[0].files == ["file1.py"]
        assert state.tasks[1].id == "task-2"
        assert state.tasks[1].description == "Second task"
        assert state.tasks[1].files == ["file2.py"]
        # Status should be preserved from TaskModel (default "pending")
        assert state.tasks[0].status == "pending"
        assert state.tasks[1].status == "pending"

    def test_load_preserves_task_fields(
        self, state_manager: StateManager, sample_run_state: RunState
    ) -> None:
        """All TaskState fields are preserved on roundtrip."""
        # Add all possible fields
        sample_run_state.tasks[0].status = "completed"
        sample_run_state.tasks[0].result = "Success output"
        sample_run_state.tasks[1].status = "failed"
        sample_run_state.tasks[1].result = "Error message"

        state_manager.save(sample_run_state)
        loaded = state_manager.load()

        assert loaded is not None
        assert loaded.tasks[0].id == "task-1"
        assert loaded.tasks[0].description == "First task"
        assert loaded.tasks[0].files == ["file1.py"]
        assert loaded.tasks[0].status == "completed"
        assert loaded.tasks[0].result == "Success output"
        assert loaded.tasks[1].id == "task-2"
        assert loaded.tasks[1].description == "Second task"
        assert loaded.tasks[1].files == ["file2.py"]
        assert loaded.tasks[1].status == "failed"
        assert loaded.tasks[1].result == "Error message"

    def test_save_idempotent(self, state_manager: StateManager, sample_run_state: RunState) -> None:
        """Saving twice produces same content (except timestamp)."""
        state_manager.save(sample_run_state)
        content1 = state_manager.path.read_text(encoding="utf-8")
        data1 = json.loads(content1)

        state_manager.save(sample_run_state)
        content2 = state_manager.path.read_text(encoding="utf-8")
        data2 = json.loads(content2)

        # Everything except updated_at should be identical
        assert data1["goal"] == data2["goal"]
        assert data1["cycles"] == data2["cycles"]
        assert data1["tasks"] == data2["tasks"]
        assert data1["history"] == data2["history"]
        assert data1["created_at"] == data2["created_at"]
        # updated_at will differ
        assert data1["updated_at"] != data2["updated_at"]