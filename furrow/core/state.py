from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.config import TaskModel


class StateManager:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.state_file = workspace / ".furrow_state.json"

    def save(self, goal: str, cycles: int, all_tasks: list[TaskModel], completed_task_ids: set[str]) -> None:
        data = {
            "goal": goal,
            "cycles": cycles,
            "tasks": [task.model_dump() for task in all_tasks],
            "completed_task_ids": list(completed_task_ids),
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def load(self) -> dict[str, Any] | None:
        if not self.state_file.exists():
            return None
        try:
            return json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def clear(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()