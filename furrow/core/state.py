"""JSON-backed persistence for goal / task / cycle state across sessions.

Standard library + pydantic only. Importing this module performs no filesystem
access; every disk interaction happens inside a :class:`StateStore` method and
is best-effort -- a missing or corrupt state file must never crash a run.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from furrow.config import Settings, TaskModel


class CycleRecord(BaseModel):
    """Outcome of a single plan -> execute -> test cycle."""

    cycle: int
    passed: bool
    summary: str = ""
    failures: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    completed: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)


class RunState(BaseModel):
    """Everything worth remembering between cycles and between sessions."""

    goal: str
    cycles: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    history: list[CycleRecord] = Field(default_factory=list)
    updated_at: str = ""  # ISO-8601 UTC timestamp, set on every save


class StateStore:
    """Reads and writes a :class:`RunState` as JSON at ``settings.state_file``.

    Defaults to ``<workspace>/.furrow/state.json`` when ``state_file`` is unset;
    the path is resolved per call so a ``workspace`` overridden after settings
    construction is honoured.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def path(self) -> Path:
        """Resolve the state file location."""
        configured: Path | None = self.settings.state_file
        if configured is not None:
            return Path(configured)
        return Path(self.settings.workspace) / ".furrow" / "state.json"

    def load(self) -> RunState | None:
        """Return the persisted state, or ``None`` if absent or unreadable."""
        target = self.path()
        try:
            raw = target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            return RunState.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError):
            return None

    def save(self, state: RunState) -> None:
        """Persist ``state`` atomically; failures are swallowed (best-effort)."""
        state.updated_at = datetime.now(timezone.utc).isoformat()
        target = self.path()
        tmp_path: str | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = state.model_dump_json(indent=2)
            fd, tmp_path = tempfile.mkstemp(
                dir=str(target.parent), prefix=target.name, suffix=".tmp"
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
            os.replace(tmp_path, target)
            tmp_path = None
        except OSError:
            pass
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    def clear(self) -> None:
        """Delete the state file if it exists; a missing file is not an error."""
        try:
            self.path().unlink()
        except OSError:
            pass
