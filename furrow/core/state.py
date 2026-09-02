"""State persistence for Furrow sessions.

This module provides the `FurrowState` model and `StateStore` class, which
allow orchestrator sessions to survive crashes and be resumed later. State is
serialized to a JSON file and written atomically to avoid corruption.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from furrow.config import Plan, TestResult
from pydantic import BaseModel


class FurrowState(BaseModel):
    """Snapshot of an orchestrator's progress.

    Captures the information needed to resume a Furrow session after a crash
    or interruption, including the goal, completed cycle count, current plan,
    last test result, and the reason (if any) the session terminated.
    """

    version: int = 1
    goal: str
    cycles: int = 0
    current_plan: Plan | None = None
    last_test_result: TestResult | None = None
    done_reason: str | None = None
    created_at: str
    updated_at: str


class StateStore:
    """Persists and loads `FurrowState` snapshots to a JSON file.

    Writes are performed atomically by writing to a temporary file and then
    replacing the destination, ensuring the on-disk state is never left in a
    partially-written (corrupt) state.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, state: FurrowState) -> None:
        """Atomically write the given state to disk, refreshing `updated_at`."""
        state.updated_at = datetime.now(timezone.utc).isoformat()
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def load(self) -> FurrowState | None:
        """Load and return persisted state, or `None` if no state file exists."""
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return FurrowState.model_validate(data)

    def clear(self) -> None:
        """Remove the state file if it exists."""
        if self.path.exists():
            self.path.unlink()

    def exists(self) -> bool:
        """Return whether a state file currently exists on disk."""
        return self.path.exists()
