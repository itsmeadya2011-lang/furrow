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
    last_passed: bool | None = None
    last_skipped: bool = False
    last_summary: str = ""


def state_path_for(
    workspace: str | Path, relative: str | Path = DEFAULT_STATE_RELATIVE_PATH
) -> Path:
    """Absolute-ish path of the state file for a workspace: workspace / relative."""
    return Path(workspace) / Path(relative)


class StateStore:
    """Persistent storage for orchestrator run state."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def exists(self) -> bool:
        """Check if a state file exists."""
        return self.path.exists()

    def load(self) -> RunState | None:
        """Load the run state from disk."""
        try:
            if self.path.exists():
                with open(self.path) as f:
                    data = json.load(f)
                return RunState(**data)
        except (json.JSONDecodeError, ValidationError):
            logger.warning("Ignoring corrupt state file %s", self.path)
        except OSError as exc:
            logger.warning("Could not read state file %s: %s", self.path, exc)
        return None

    def save(self, state: RunState) -> None:
        """Save the run state to disk atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                json.dump(state.dict(), f, indent=2)
            temp_path.replace(self.path)
        except OSError as e:
            logger.error("Failed to save state to %s: %s", self.path, e)
            raise

    def clear(self) -> None:
        """Remove the state file if it exists."""
        try:
            if self.path.exists():
                self.path.unlink()
        except OSError as e:
            logger.warning("Failed to clear state file %s: %s", self.path, e)
