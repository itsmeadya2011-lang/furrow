from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.config import TaskModel, TestResult

STATE_FILENAME = "furrow_state.json"


def _serialize(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable values."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]
    return obj


class StateStore:
    """Persists orchestrator state to a JSON file for cross-session tracking.

    The state file lives in the workspace root (``furrow_state.json`` by
    default) and survives process restarts so a later run can resume.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else Path(STATE_FILENAME)

    def save(self, data: dict[str, Any]) -> None:
        payload = _serialize(data)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str))
        tmp.replace(self.path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return {}

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass


def state_from_dict(data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a loaded JSON dict, converting task/test records to typed objects."""
    if not data:
        return {
            "goal": "",
            "cycles": 0,
            "tasks": [],
            "last_test_result": None,
            "done": False,
        }
    tasks_raw = data.get("tasks", []) or []
    tasks = [TaskModel(**t) if isinstance(t, dict) else t for t in tasks_raw]
    test_result_raw = data.get("last_test_result")
    test_result = TestResult(**test_result_raw) if test_result_raw else None
    return {
        "goal": data.get("goal", ""),
        "cycles": data.get("cycles", 0) or 0,
        "tasks": tasks,
        "last_test_result": test_result,
        "done": bool(data.get("done", False)),
    }
