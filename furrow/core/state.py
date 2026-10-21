"""Persistent run state for Furrow.

The orchestrator keeps its goal, tasks and cycle history in RAM only, so a
crash or a restart loses every bit of progress. This module provides a small,
dependency-free persistence layer that stores the run state as JSON on disk.

Design notes:
  * Writes are atomic: the payload is written to a temporary file in the same
    directory, flushed and fsync'ed, then moved into place with ``os.replace``.
    A partially written state file can therefore never be observed.
  * Reading is forgiving: a missing or corrupt state file yields ``None``
    instead of raising, so a run can always start fresh.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from furrow.config import TaskModel, Usage
from furrow.logging import get_logger

logger = get_logger(__name__)

STATE_VERSION = 1

DEFAULT_STATE_RELATIVE_PATH = Path(".furrow/state.json")


def utcnow() -> str:
    """ISO-8601 UTC timestamp string, e.g. 2026-10-20T12:30:04.395123+00:00"""
    return datetime.now(timezone.utc).isoformat()


class HistoryEntry(BaseModel):
    """Summary of a single orchestration cycle."""

    cycle: int = 0
    goal: str = ""
    plan_rationale: str = ""
    task_summaries: list[str] = Field(default_factory=list)
    tests_passed: bool | None = None
    test_summary: str = ""
    timestamp: str = Field(default_factory=utcnow)
    usage: Usage = Field(default_factory=Usage)


class RunState(BaseModel):
    """Everything worth surviving a restart."""

    version: int = STATE_VERSION
    goal: str = ""
    original_goal: str = ""
    cycles: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    history: list[HistoryEntry] = Field(default_factory=list)
    updated_at: str = Field(default_factory=utcnow)
    usage: Usage = Field(default_factory=Usage)
    no_progress: int = 0
    prev_sig: str = ""
    tests_run: int = 0

def state_path_for(
    workspace: str | Path, relative: str | Path = DEFAULT_STATE_RELATIVE_PATH
) -> Path:
    """Absolute-ish path of the state file for a workspace: workspace / relative."""
    return Path(workspace) / Path(relative)


class StateStore:
    """Reads and writes a :class:`RunState` JSON document."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        """True when the state file is present on disk."""
        return self.path.is_file()

    def load(self) -> RunState | None:
        """Return the saved RunState, or None when the file is missing or corrupt.

        A corrupt file must NOT raise - log a warning via the `logging` module and
        return None so a run can start fresh instead of crashing.
        """
        if not self.path.is_file():
            return None
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not read state file %s: %s", self.path, exc)
            return None
        try:
            return RunState.model_validate_json(raw)
        except (ValidationError, ValueError) as exc:
            logger.warning("Ignoring corrupt state file %s: %s", self.path, exc)
            return None

    def save(self, state: RunState) -> None:
        """Serialize state to JSON and write it ATOMICALLY.

        The payload is written to a temporary file in the same directory, then
        flushed, fsync'ed and moved into place with ``os.replace``. If anything
        fails the temporary file is removed and the error is re-raised.
        """
        state.updated_at = utcnow()
        payload = json.dumps(state.model_dump(mode="json"), indent=2)

        parent = self.path.parent
        parent.mkdir(parents=True, exist_ok=True)

        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                delete=False,
                dir=parent,
                prefix=f"{self.path.name}.",
                suffix=".tmp",
            ) as tmp:
                tmp_name = tmp.name
                tmp.write(payload)
                tmp.write("\n")
                tmp.flush()
                os.fsync(tmp.fileno())
            os.replace(tmp_name, self.path)
            tmp_name = None
        finally:
            if tmp_name is not None:
                try:
                    os.unlink(tmp_name)
                except OSError:  # pragma: no cover - best effort cleanup
                    logger.warning("Could not remove temporary file %s", tmp_name)

    def clear(self) -> None:
        """Delete the state file if it exists; never raise when it is already missing."""
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Could not remove state file %s: %s", self.path, exc)
