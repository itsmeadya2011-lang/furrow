from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.config import Plan, TaskModel
from furrow.logging import get_logger

logger = get_logger(__name__)

STATE_FILE = Path(".furrow_state.json")


class StateManager:
    """Manages persistent state for Furrow across sessions."""
    
    def __init__(self, state_file: Path = STATE_FILE) -> None:
        self.state_file = state_file
    
    def save_state(self, goal: str, tasks: list[TaskModel], cycles: int) -> None:
        """Save the current state to disk."""
        state = {
            "goal": goal,
            "tasks": [task.model_dump() for task in tasks],
            "cycles": cycles,
        }
        try:
            self.state_file.write_text(json.dumps(state, indent=2))
            logger.debug("state_saved", path=str(self.state_file))
        except Exception as e:
            logger.error("state_save_failed", error=str(e))
    
    def load_state(self) -> dict[str, Any] | None:
        """Load state from disk."""
        if not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text())
            logger.info("state_loaded", path=str(self.state_file))
            return data
        except Exception as e:
            logger.error("state_load_failed", error=str(e))
            return None
    
    def clear_state(self) -> None:
        """Clear the saved state."""
        if self.state_file.exists():
            self.state_file.unlink()
            logger.info("state_cleared")
    
    def restore_tasks(self, data: dict[str, Any]) -> list[TaskModel]:
        """Restore TaskModel objects from saved state."""
        tasks_data = data.get("tasks", [])
        return [TaskModel(**task_data) for task_data in tasks_data]