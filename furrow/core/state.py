"""Persistent state management for the Furrow orchestrator.

State is stored as a JSON file (default ``.furrow-state.json`` in the
workspace root) so that progress survives process restarts or across
sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from furrow.core.orchestrator import Orchestrator

logger = structlog.get_logger()

DEFAULT_STATE_FILE = ".furrow-state.json"


def save_state(orchestrator: Orchestrator, path: str | Path) -> None:
    """Serialize the orchestrator's tracked state to ``path``."""
    state: dict[str, Any] = {
        "goal": orchestrator.goal,
        "cycles": orchestrator.cycles,
        "tasks": [t.model_dump() for t in orchestrator.tasks],
        "max_cycles": orchestrator.max_cycles,
        "max_parallel_tasks": orchestrator.max_parallel_tasks,
    }
    if orchestrator.last_test_result is not None:
        state["last_test_result"] = orchestrator.last_test_result.model_dump()

    p = Path(path)
    p.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    logger.info("state_saved", path=str(p), cycles=orchestrator.cycles)


def load_state(path: str | Path) -> dict[str, Any] | None:
    """Load state from ``path`` if it exists.

    Returns ``None`` if the file does not exist or is unreadable.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        logger.info("state_loaded", path=str(p), cycles=data.get("cycles", 0))
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("state_load_failed", path=str(p), error=str(e))
        return None


def clear_state(path: str | Path) -> None:
    """Remove the state file if it exists."""
    p = Path(path)
    if p.exists():
        p.unlink()
        logger.info("state_cleared", path=str(p))
