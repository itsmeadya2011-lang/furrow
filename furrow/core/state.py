from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STATE_DIR_NAME = ".furrow"
STATE_FILE_NAME = "state.json"


def state_dir(workspace: Path | None = None) -> Path:
    """Return the state directory (workspace/.furrow), created on demand by save_state."""
    return (workspace or Path.cwd()) / STATE_DIR_NAME


def state_path(workspace: Path | None = None) -> Path:
    """Return the state file path (workspace/.furrow/state.json)."""
    return state_dir(workspace) / STATE_FILE_NAME


def save_state(state: dict[str, Any], workspace: Path | None = None) -> Path:
    """Atomically write state as pretty JSON to the state file.

    Creates the state directory if needed, writes to a temp file in the same
    directory, then os.replace()s it into place. Adds/updates an 'updated_at'
    ISO-8601 UTC timestamp field before writing. Returns the state file path.
    """
    path = state_path(workspace)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(state)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, delete=False, suffix=".tmp"
        ) as f:
            tmp = f.name
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
        return path
    except Exception:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass
        raise


def load_state(workspace: Path | None = None) -> dict[str, Any] | None:
    """Load the persisted state, or None if missing or corrupt."""
    path = state_path(workspace)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def clear_state(workspace: Path | None = None) -> bool:
    """Delete the state file if it exists. Return True if a file was removed."""
    path = state_path(workspace)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True