"""Persistent run state for Furrow.

The orchestrator keeps its goal, tasks and cycle history in RAM, but that state
is lost on a crash or restart. This module provides a small, dependency-free
persistence layer that stores the run state as JSON on disk so progress survives
restarts.

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
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from furrow.config import TaskModel, Usage
from furrow.logging import get_logger

logger = get_logger(__name__)

STATE_VERSION = 2

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


def compact_history(state: RunState, max_entries: int) -> RunState:
    """Return a copy of *state* with its history compacted to *max_entries*.

    When ``len(state.history) > max_entries`` the oldest entries are replaced
    with a single summary entry that records the range of compacted cycles and
    aggregates their usage. This bounds the size of the persisted state file.
    """
    if max_entries <= 0 or len(state.history) <= max_entries:
        return state
    # Number of entries to keep from the end (most recent)
    keep = max_entries - 1  # Reserve 1 slot for the summary entry
    if keep <= 0:
        keep = 1
    to_compact = state.history[:-keep]
    kept = state.history[-keep:]

    # Aggregate usage from compacted entries
    compacted_usage = Usage()
    for entry in to_compact:
        compacted_usage.add(entry.usage)

    summary_entry = HistoryEntry(
        cycle=state.cycles,
        goal="",
        plan_rationale=(
            f"History compacted: {len(to_compact)} cycles "
            f"({to_compact[0].cycle}-{to_compact[-1].cycle})"
        ),
        task_summaries=[],
        tests_passed=None,
        test_summary=f"Compacted {len(to_compact)} cycle(s)",
        timestamp=utcnow(),
        usage=compacted_usage,
    )
    compacted_history = [summary_entry, *kept]
    # Return a new RunState with compacted history
    return RunState(
        version=state.version,
        goal=state.goal,
        original_goal=state.original_goal,
        cycles=state.cycles,
        tasks=state.tasks,
        history=compacted_history,
        updated_at=state.updated_at,
        usage=state.usage,
        no_progress=state.no_progress,
        prev_sig=state.prev_sig,
        tests_run=state.tests_run,
        last_passed=state.last_passed,
        last_skipped=state.last_skipped,
        last_summary=state.last_summary,
    )


def _migrate_v1_to_v2(data: dict) -> dict:
    """Migrate state from version 1 to version 2.

    Version 2 adds:
    - max_history_entries field to Settings (handled by config, not state)
    - compacted history support (handled by compact_history function)
    - cost budget fields (handled by config, not state)

    The state schema itself didn't change between v1 and v2, but we increment
    the version to track the schema version for future migrations.
    """
    # No schema changes needed for v1 -> v2, just ensure version is set
    data["version"] = STATE_VERSION
    return data


_MIGRATIONS: dict[int, Callable[[dict], dict]] = {
    1: _migrate_v1_to_v2,
}


def migrate_state(data: dict) -> dict:
    """Apply migrations to bring state data to the current version.

    Returns the migrated data dict, or the original if no migration needed.
    If the data version is newer than STATE_VERSION, returns the data as-is
    (forward compatibility).
    """
    version = data.get("version", 1)
    if version >= STATE_VERSION:
        return data
    for v in range(version, STATE_VERSION):
        migrator = _MIGRATIONS.get(v)
        if migrator:
            data = migrator(data)
        else:
            logger.warning("No migration from v%d to v%d, skipping", v, v + 1)
    return data


def _fsync_directory(directory: Path) -> None:
    """fsync the parent directory so a rename landing in it is durable.

    On POSIX, ``os.replace`` only makes the new directory entry visible; the
    data is not durable until the directory itself is fsynced. A crash right
    after :meth:`StateStore.save` could otherwise lose the state file even
    though the temporary payload was fsynced.

    Durability hardening must never make a save fail, so any filesystem that
    cannot fsync a directory (or any ``OSError`` while opening/fsyncing it) is
    logged at warning/debug level and swallowed. On non-POSIX platforms this
    is a no-op.
    """
    if os.name != "posix":
        return
    fd = -1
    try:
        try:
            fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError as exc:
            logger.debug("Could not open directory %s for fsync: %s", directory, exc)
            return
        try:
            os.fsync(fd)
        except OSError as exc:
            logger.warning("Could not fsync directory %s: %s", directory, exc)
    finally:
        if fd >= 0:
            os.close(fd)


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
        except (OSError, UnicodeError) as exc:
            logger.warning("Could not read state file %s: %s", self.path, exc)
            return None
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("state root must be a JSON object")
            data = migrate_state(data)
            return RunState.model_validate(data)
        except (ValidationError, ValueError, RecursionError) as exc:
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
            _fsync_directory(self.path.parent)
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
