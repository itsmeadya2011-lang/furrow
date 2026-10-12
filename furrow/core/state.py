from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.config import Settings, TaskModel, settings


class StateManager:
    def __init__(self, workspace: Path | None = None, settings: Settings | None = None) -> None:
        self.settings = settings or settings
        self.workspace = workspace or self.settings.workspace
        self.state_file = self.workspace / ".furrow_state.json"

    def save(self, goal: str, all_tasks: list[TaskModel], cycle_history: list[dict[str, Any]], cycles: int) -> None:
        """Save orchestrator state to disk."""
        state = {
            "goal": goal,
            "cycles": cycles,
            "all_tasks": [t.model_dump() for t in all_tasks],
            "cycle_history": cycle_history,
        }
        self.state_file.write_text(json.dumps(state, indent=2))

    def load(self) -> dict[str, Any] | None:
        """Load orchestrator state from disk."""
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