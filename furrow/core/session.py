from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from furrow.config import SessionState

logger = logging.getLogger(__name__)


class SessionError(Exception):
    """Base exception for session persistence errors."""


class SessionNotFoundError(SessionError):
    """Raised when a requested session does not exist."""


class SessionCorruptedError(SessionError):
    """Raised when a session file cannot be parsed."""


def generate_session_id() -> str:
    """Generate a new session identifier.

    Combines a timestamp prefix (for human-friendly ordering) with a
    short UUID suffix for uniqueness.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


class SessionManager:
    """Persist and load Furrow orchestration sessions on disk.

    Sessions are stored as JSON files in ``<workspace>/.furrow/sessions/``.
    Each session is a single file named ``<session_id>.json``.
    """

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace)
        self.sessions_dir = self.workspace / ".furrow" / "sessions"

    @property
    def dir(self) -> Path:
        return self.sessions_dir

    def _ensure_dir(self) -> None:
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise SessionError(
                f"Failed to create sessions directory {self.sessions_dir}: {exc}"
            ) from exc

    def _path_for(self, session_id: str) -> Path:
        # Sanitize to prevent path traversal: only allow safe characters.
        if not session_id or any(c in session_id for c in ("/", "\\", "..", "\0")):
            raise SessionError(f"Invalid session id: {session_id!r}")
        return self.sessions_dir / f"{session_id}.json"

    def save(self, session_id: str, state: SessionState) -> Path:
        """Persist ``state`` under ``session_id``.

        Returns the path of the written file.
        """
        self._ensure_dir()
        path = self._path_for(session_id)
        state.session_id = session_id
        state.touch()
        try:
            data = state.to_json()
            # Write to a temp file first then atomically replace, so a crash
            # mid-write cannot corrupt the existing session.
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            tmp_path.write_text(data, encoding="utf-8")
            tmp_path.replace(path)
        except OSError as exc:
            raise SessionError(f"Failed to write session {session_id}: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise SessionError(f"Failed to serialize session {session_id}: {exc}") from exc
        logger.debug("Saved session %s to %s", session_id, path)
        return path

    def load(self, session_id: str) -> SessionState:
        """Load a session by id.

        Raises ``SessionNotFoundError`` if the session does not exist and
        ``SessionCorruptedError`` if the file is unreadable or invalid.
        """
        path = self._path_for(session_id)
        if not path.exists():
            raise SessionNotFoundError(f"Session not found: {session_id}")
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SessionCorruptedError(
                f"Failed to read session file {path}: {exc}"
            ) from exc
        try:
            return SessionState.from_json(raw)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise SessionCorruptedError(
                f"Session file is corrupted and cannot be parsed: {path}"
            ) from exc

    def exists(self, session_id: str) -> bool:
        try:
            return self._path_for(session_id).exists()
        except SessionError:
            return False

    def list_sessions(self) -> list[SessionState]:
        """Return all sessions, sorted by ``created_at`` (oldest first)."""
        self._ensure_dir()
        results: list[SessionState] = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                raw = path.read_text(encoding="utf-8")
                state = SessionState.from_json(raw)
            except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
                logger.warning("Skipping corrupted session file %s: %s", path, exc)
                continue
            results.append(state)
        results.sort(key=lambda s: s.created_at)
        return results

    def delete(self, session_id: str) -> bool:
        """Delete the session file. Returns ``True`` if a file was removed."""
        try:
            path = self._path_for(session_id)
        except SessionError:
            return False
        try:
            path.unlink()
            logger.debug("Deleted session %s", session_id)
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise SessionError(f"Failed to delete session {session_id}: {exc}") from exc

    def new_session(
        self,
        goal: str,
        workspace: Optional[Path | str] = None,
        session_id: Optional[str] = None,
    ) -> tuple[str, SessionState]:
        """Create a new session and persist it.

        Returns a tuple of ``(session_id, state)``.
        """
        sid = session_id or generate_session_id()
        ws = Path(workspace) if workspace is not None else self.workspace
        state = SessionState(
            session_id=sid,
            goal=goal,
            current_goal=goal,
            cycles=0,
            current_plan=None,
            status="running",
            workspace=str(ws),
        )
        self.save(sid, state)
        return sid, state
