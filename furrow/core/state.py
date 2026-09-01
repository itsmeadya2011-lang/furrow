from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from furrow.config import settings


class StateManager:
    """Manages persistent state for Furrow sessions."""

    def __init__(self, state_file: Path | None = None) -> None:
        self.state_file = state_file or Path(settings.workspace) / ".furrow_state.json"
        self._state: dict[str, Any] = self._load()

    def _load(self) -> dict[str, Any]:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                return self._default_state()
        return self._default_state()

    def _default_state(self) -> dict[str, Any]:
        """Return default empty state."""
        return {
            "version": "1.0",
            "sessions": [],
            "current_session": None,
        }

    def save(self) -> None:
        """Save current state to file."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    def start_session(self, goal: str) -> str:
        """Start a new session and return its ID."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = {
            "id": session_id,
            "goal": goal,
            "started_at": datetime.now().isoformat(),
            "completed_at": None,
            "cycles": 0,
            "status": "running",
            "tasks_completed": [],
            "tasks_failed": [],
        }
        self._state["sessions"].append(session)
        self._state["current_session"] = session_id
        self.save()
        return session_id

    def update_session(self, session_id: str, **kwargs: Any) -> None:
        """Update session data."""
        for session in self._state["sessions"]:
            if session["id"] == session_id:
                session.update(kwargs)
                self.save()
                return

    def complete_session(self, session_id: str, status: str = "completed") -> None:
        """Mark a session as complete."""
        for session in self._state["sessions"]:
            if session["id"] == session_id:
                session["status"] = status
                session["completed_at"] = datetime.now().isoformat()
                self.save()
                return

    def add_task_result(self, session_id: str, task_id: str, status: str, description: str) -> None:
        """Record a task result."""
        for session in self._state["sessions"]:
            if session["id"] == session_id:
                if status == "completed":
                    session["tasks_completed"].append({
                        "id": task_id,
                        "description": description,
                        "completed_at": datetime.now().isoformat(),
                    })
                elif status == "failed":
                    session["tasks_failed"].append({
                        "id": task_id,
                        "description": description,
                        "failed_at": datetime.now().isoformat(),
                    })
                session["cycles"] = session.get("cycles", 0) + 1
                self.save()
                return

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        """Get session by ID."""
        for session in self._state["sessions"]:
            if session["id"] == session_id:
                return session
        return None

    def get_current_session(self) -> dict[str, Any] | None:
        """Get the current active session."""
        if self._state["current_session"]:
            return self.get_session(self._state["current_session"])
        return None

    def get_all_sessions(self) -> list[dict[str, Any]]:
        """Get all sessions."""
        return self._state["sessions"]

    def clear_history(self) -> None:
        """Clear all session history."""
        self._state = self._default_state()
        self.save()
