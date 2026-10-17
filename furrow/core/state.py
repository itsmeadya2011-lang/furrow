from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.config import TaskModel

STATE_FILE = Path(".furrow_state.json")


def save_state(orchestrator: Any, path: str | Path) -> None:
    """Serialize goal, cycles, and tasks to a JSON file."""
    path = Path(path)
    state = {
        "goal": orchestrator.goal,
        "cycles": orchestrator.cycles,
        "tasks": [t.model_dump() for t in orchestrator._get_tasks()],
    }
    path.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def load_state(path: str | Path) -> dict | None:
    """Load state dict from JSON file, or None if missing/invalid."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None