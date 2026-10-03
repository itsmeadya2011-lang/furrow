"""Durable run state for the orchestrator loop.

State is written as a small JSON document so a run can be inspected (or resumed)
without importing furrow. Every read is defensive: a missing or corrupt file is
reported as "no state" rather than an exception, because a broken state file must
never be able to stop the loop.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from furrow.config import TaskModel


class RunState(BaseModel):
    """Serializable snapshot of one orchestrator run."""

    goal: str
    fix_instructions: str = ""
    cycles: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    done: bool = False
    halted_reason: str | None = None
    history: list[dict[str, Any]] = Field(default_factory=list)


class StateStore:
    """JSON file backed store for :class:`RunState`."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> RunState | None:
        """Return the persisted state, or ``None`` when missing/unreadable/corrupt."""
        try:
            raw = self._path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        try:
            return RunState.model_validate_json(raw)
        except (ValidationError, ValueError, TypeError):
            return None

    def save(self, state: RunState) -> None:
        """Atomically write ``state``: temp file in the same dir + ``os.replace``."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(state.model_dump(mode="json"), indent=2, default=str)
        handle_fd, temp_name = tempfile.mkstemp(
            dir=str(self._path.parent), prefix=f".{self._path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self._path)
        except BaseException:
            with suppress(OSError):
                os.unlink(temp_name)
            raise

    def clear(self) -> None:
        """Delete the state file if it exists. Missing files are not an error."""
        with suppress(OSError):
            self._path.unlink()