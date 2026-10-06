from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from furrow.config import TaskModel


class FurrowState(BaseModel):
    goal: str
    cycles: int = 0
    last_tasks: list[dict[str, Any]] = []


class StateManager:
    def __init__(self, workspace: Path | str | None = None) -> None:
        self.workspace = Path(workspace) if workspace else Path.cwd()
        self.state_dir = self.workspace / ".furrow"
        self.state_file = self.state_dir / "state.json"

    def load(self) -> FurrowState | None:
        if not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text())
            return FurrowState(**data)
        except (json.JSONDecodeError, ValueError):
            return None

    def save(self, state: FurrowState) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(state.model_dump(), indent=2))
