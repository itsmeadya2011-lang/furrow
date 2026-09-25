from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from furrow.config import Settings, settings


STATE_FILE = ".furrow_state.json"


def get_state_path(workspace: Path | None = None) -> Path:
    """Get the path to the state file."""
    ws = workspace or settings.workspace
    return ws / STATE_FILE


def save_state(orchestrator: Any, workspace: Path | None = None) -> None:
    """Save orchestrator state to file."""
    state_path = get_state_path(workspace)
    state = orchestrator.get_state()
    state_path.write_text(json.dumps(state, indent=2))


def load_state(workspace: Path | None = None) -> dict[str, Any] | None:
    """Load orchestrator state from file."""
    state_path = get_state_path(workspace)
    if not state_path.exists():
        return None
    try:
        return json.loads(state_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def clear_state(workspace: Path | None = None) -> None:
    """Clear saved state."""
    state_path = get_state_path(workspace)
    if state_path.exists():
        state_path.unlink()