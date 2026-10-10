from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.config import TaskModel


class StateManager:
    """Manages persistent state for Furrow across runs."""

    def __init__(self, workspace: Path, goal: str) -> None:
        self.workspace = workspace
        self.goal = goal
        self.state_file = workspace / ".furrow_state.json"

    def save(self, tasks: list[TaskModel], cycles: int, current_goal: str) -> None:
        """Save current state to disk."""
        state = {
            "goal": current_goal,
            "original_goal": self.goal,
            "cycles": cycles,
            "tasks": [task.model_dump() for task in tasks],
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def load(self) -> dict[str, Any] | None:
        """Load state from disk if it exists."""
        if not self.state_file.exists():
            return None
        try:
            return json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self) -> None:
        """Clear saved state."""
        if self.state_file.exists():
            self.state_file.unlink()

    def restore_tasks(self, data: dict[str, Any]) -> list[TaskModel]:
        """Restore TaskModel objects from saved state."""
        return [TaskModel(**task_data) for task_data in data.get("tasks", [])]