from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from furrow.config import TaskModel


class StateManager:
    """Manages persistent state for Furrow runs."""
    
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.state_file = workspace / ".furrow_state.json"
        self._state: dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    self._state = json.load(f)
            except (json.JSONDecodeError, OSError):
                self._state = {}

    def _save(self) -> None:
        """Save state to file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self._state, f, indent=2, default=str)
        except OSError:
            pass

    def save_tasks(self, tasks: list[TaskModel]) -> None:
        """Save task list to state."""
        self._state["tasks"] = [task.model_dump() for task in tasks]
        self._state["updated_at"] = time.time()
        self._save()

    def load_tasks(self) -> list[TaskModel]:
        """Load task list from state."""
        tasks_data = self._state.get("tasks", [])
        return [TaskModel(**t) for t in tasks_data]

    def save_goal(self, goal: str) -> None:
        """Save current goal."""
        self._state["goal"] = goal
        self._save()

    def load_goal(self) -> str | None:
        """Load saved goal."""
        return self._state.get("goal")

    def save_cycle(self, cycle: int) -> None:
        """Save current cycle number."""
        self._state["cycle"] = cycle
        self._save()

    def load_cycle(self) -> int:
        """Load saved cycle number."""
        return self._state.get("cycle", 0)

    def clear(self) -> None:
        """Clear all state."""
        self._state = {}
        if self.state_file.exists():
            self.state_file.unlink()

    def get_state(self) -> dict[str, Any]:
        """Get full state dict."""
        return self._state.copy()