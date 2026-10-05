"""Crash-safe JSON persistence for furrow runs.

The orchestrator checkpoints its goal, task table and last test outcome into
one small JSON document per run so that a crash (or a ``Ctrl-C``) does not lose
the whole loop and so the CLI can list past runs.

Guarantees:

* **Atomic writes** -- a record is written to ``<name>.json.tmp``, flushed and
  then moved into place with :func:`os.replace`, so a reader never observes a
  half-written file and a crash can never corrupt an existing record.
* **No I/O exceptions** -- a missing, unreadable or hand-edited document is
  reported as "no such run" instead of raising.  Losing a checkpoint must never
  take down a run.  The one deliberate exception is ``ValueError`` for an
  unsafe ``run_id`` (path-traversal guard, see :func:`_validate_run_id`).
* **Side-effect free construction** -- :class:`RunStore` only creates directories
  on the first :meth:`RunStore.save`.

Layout::

    <directory>/runs/<run_id>.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

__all__ = ["RunRecord", "RunStatus", "RunStore", "summarize"]

#: A run id is used verbatim as a filename, so it must be a single, boring
#: path segment: letters, digits, dot, underscore and dash only.
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9._-]+$")
_MAX_RUN_ID_LEN = 128
_GOAL_SUMMARY_CHARS = 60


class RunStatus(str, Enum):
    """Terminal (or in-flight) state of a run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    MAX_CYCLES = "max_cycles"
    STAGNATED = "stagnated"


class RunRecord(BaseModel):
    """Everything needed to inspect -- and restart -- one furrow run.

    Attributes:
        run_id: Filesystem-safe identifier, also the JSON filename.
        goal: The goal handed to the orchestrator (possibly rewritten after a
            failing test cycle).
        status: Current :class:`RunStatus`.
        cycles: Number of planner/worker/tester cycles executed so far.
        created_at: Epoch seconds when the run record was created.
        updated_at: Epoch seconds of the most recent :meth:`RunStore.save`.
        tasks: Snapshots of ``TaskModel.model_dump()``, one dict per task.
        last_test: Snapshot of the last ``TestResult.model_dump()`` or ``None``.
        last_failures: Failure lines from the last failing test run.
        summary: Free-form terminal summary (e.g. ``{"cycles": 3}``).
        notes: Human notes, e.g. why the loop halted.
    """

    run_id: str
    goal: str
    status: RunStatus = RunStatus.PENDING
    cycles: int = 0
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    tasks: list[dict[str, Any]] = Field(default_factory=list)
    last_test: dict[str, Any] | None = None
    last_failures: list[str] = Field(default_factory=list)
    summary: dict[str, Any] | None = None
    notes: str = ""


def summarize(record: RunRecord) -> str:
    """Render a single-line, human readable summary of ``record``.

    Example::

        20260105T101500-9f2c1a  completed  3 cycles  4/5 tasks  Build the thing
    """
    tasks = record.tasks or []
    done = sum(1 for task in tasks if isinstance(task, dict) and task.get("status") == "completed")
    status = getattr(record.status, "value", record.status)
    goal = " ".join((record.goal or "").split())
    if len(goal) > _GOAL_SUMMARY_CHARS:
        goal = goal[:_GOAL_SUMMARY_CHARS].rstrip() + "..."
    return f"{record.run_id}  {status}  {record.cycles} cycles  {done}/{len(tasks)} tasks  {goal}"


def _validate_run_id(run_id: str) -> str:
    """Return ``run_id`` unchanged, or raise ``ValueError`` when it is unsafe.

    The identifier becomes a filename, so ``..``, ``a/b``, ``a\\b``, NUL bytes,
    shell metacharacters and absurdly long names are all rejected.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if len(run_id) > _MAX_RUN_ID_LEN:
        raise ValueError(f"run_id is too long (max {_MAX_RUN_ID_LEN} chars): {run_id!r}")
    if not _SAFE_RUN_ID.match(run_id):
        raise ValueError(
            f"unsafe run_id {run_id!r}: only letters, digits, '.', '_' and '-' are allowed"
        )
    if set(run_id) <= {"."}:  # ".", "..", "..." -- matches the regex but is not a name
        raise ValueError(f"unsafe run_id {run_id!r}")
    return run_id


def _unlink_quietly(path: Path) -> None:
    """Delete ``path`` if present, ignoring any I/O error."""
    try:
        path.unlink()
    except OSError:
        pass


class RunStore:
    """Reads and writes :class:`RunRecord` documents under ``directory``.

    Args:
        directory: Root for persisted state; run documents live in
            ``<directory>/runs``.  Created lazily on the first save, so
            constructing a store never touches the filesystem.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory).expanduser()

    @classmethod
    def from_settings(cls, settings: Any) -> RunStore:
        """Build a store from a settings object, tolerating missing fields.

        Uses ``settings.state_dir`` when present, otherwise falls back to
        ``<settings.workspace>/.furrow`` (and finally ``.furrow``) so this keeps
        working while ``furrow.config`` is still evolving.
        """
        raw = getattr(settings, "state_dir", None)
        if not raw:
            workspace = getattr(settings, "workspace", None) or "."
            raw = Path(workspace) / ".furrow"
        return cls(raw)

    @property
    def runs_dir(self) -> Path:
        """Directory holding one ``<run_id>.json`` document per run."""
        return self.directory / "runs"

    def path_for(self, run_id: str) -> Path:
        """Return the JSON path for ``run_id``.

        Raises:
            ValueError: if ``run_id`` is not a safe filename (traversal guard).
        """
        safe = _validate_run_id(run_id)
        return self.runs_dir / f"{safe}.json"

    def save(self, record: RunRecord) -> Path:
        """Atomically persist ``record`` and return its path.

        ``updated_at`` is refreshed on every save.  I/O problems are logged and
        swallowed so a read-only or corrupt state directory cannot kill a run;
        only an unsafe ``record.run_id`` raises (``ValueError``).
        """
        path = self.path_for(record.run_id)
        record.updated_at = time.time()
        payload = json.dumps(record.model_dump(mode="json"), indent=2, default=str)
        tmp = path.with_name(path.name + ".tmp")
        try:
            self.runs_dir.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
        except (OSError, ValueError) as exc:  # persistence must never kill a run
            logger.error("could not persist run %s: %s", record.run_id, exc)
            _unlink_quietly(tmp)
        return path

    def load(self, run_id: str) -> RunRecord | None:
        """Return the stored record for ``run_id``, or ``None``.

        ``None`` means "missing, unreadable or corrupt" -- never an exception --
        so a half-written file from a previous crash is simply skipped.
        """
        return self._load_path(self.path_for(run_id))

    @staticmethod
    def _load_path(path: Path) -> RunRecord | None:
        """Parse one JSON document, returning ``None`` when it is unusable.

        Covers a missing file, a truncated one (crash mid-write on a filesystem
        without atomic replace), undecodable bytes, malformed JSON and JSON that
        no longer validates against :class:`RunRecord`.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return None
        except ValueError:  # e.g. undecodable bytes
            logger.warning("ignoring unreadable run file %s", path.name)
            return None
        try:
            return RunRecord.model_validate(json.loads(raw))
        except ValidationError as exc:
            logger.warning("ignoring invalid run file %s: %s", path.name, exc)
            return None
        except ValueError as exc:  # json.JSONDecodeError subclasses ValueError
            logger.warning("ignoring corrupt run file %s: %s", path.name, exc)
            return None

    def list_runs(self, limit: int = 20) -> list[RunRecord]:
        """Return up to ``limit`` stored runs, newest ``created_at`` first."""
        if limit <= 0:
            return []
        try:
            paths = list(self.runs_dir.glob("*.json"))
        except OSError:
            return []
        records: list[RunRecord] = []
        for path in paths:
            record = self._load_path(path)
            if record is not None:
                records.append(record)
        records.sort(key=lambda item: item.created_at, reverse=True)
        return records[:limit]

    def latest(self) -> RunRecord | None:
        """Return the most recently created run record, if any."""
        records = self.list_runs(limit=1)
        return records[0] if records else None

    def delete(self, run_id: str) -> bool:
        """Delete the stored run. Returns ``True`` when a file was removed."""
        path = self.path_for(run_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            logger.warning("could not delete run %s: %s", run_id, exc)
            return False
        _unlink_quietly(path.with_name(path.name + ".tmp"))
        return True

    @staticmethod
    def new_run_id() -> str:
        """Return a sortable, filesystem-safe run id.

        Format: ``YYYYMMDDTHHMMSS-<6 hex chars>`` (UTC), e.g.
        ``20260105T101500-9f2c1a``.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        return f"{stamp}-{secrets.token_hex(3)}"

    @staticmethod
    def summarize(record: RunRecord) -> str:
        """One-line summary of ``record``; see the module-level :func:`summarize`."""
        return summarize(record)
