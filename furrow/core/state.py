from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from furrow.config import Plan, TaskModel, TestResult, settings


class SessionStatus(str, Enum):
    """Enumeration of possible session states."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class SessionState(BaseModel):
    """Persistent state for a Furrow session."""

    goal: str
    tasks: list[TaskModel] = Field(default_factory=list)
    cycles: int = 0
    plan_history: list[dict] = Field(default_factory=list)
    test_history: list[dict] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    status: SessionStatus = SessionStatus.PENDING
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def mark_updated(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def update_tasks(self, tasks: list[TaskModel]) -> None:
        """Replace the tasks list and mark updated."""
        self.tasks = tasks
        self.mark_updated()

    def add_error(self, error: str) -> None:
        """Append an error message and mark updated."""
        self.errors.append(error)
        self.mark_updated()

    def set_status(self, status: SessionStatus) -> None:
        """Set session status and mark updated."""
        self.status = status
        self.mark_updated()


class StateManager:
    """Manages persistent session state on disk."""

    def __init__(self, state_file: str | Path | None = None) -> None:
        self.state_file = Path(state_file) if state_file else self._default_path()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _default_path() -> Path:
        """Return the default state file path: <workspace>/.furrow/state.json"""
        return Path(settings.workspace) / ".furrow" / "state.json"

    def load(self) -> Optional[SessionState]:
        """Load session state from disk. Returns None if not found or invalid."""
        if not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text())
            return SessionState(**data)
        except (json.JSONDecodeError, ValueError):
            return None

    def save(self, state: SessionState) -> None:
        """Persist session state to disk."""
        state.mark_updated()
        self.state_file.write_text(state.model_dump_json(indent=2))

    def init_state(self, goal: str) -> SessionState:
        """Initialize a new session state."""
        state = SessionState(goal=goal)
        state.status = SessionStatus.RUNNING
        self.save(state)
        return state

    def update_tasks(self, tasks: list[TaskModel]) -> None:
        """Update the tasks list in persisted state."""
        state = self.load()
        if state:
            state.update_tasks(tasks)
            self.save(state)

    def add_plan_to_history(self, plan: Plan) -> None:
        """Append a plan to the plan history."""
        state = self.load()
        if state:
            state.plan_history.append(plan.model_dump())
            self.save(state)

    def add_test_to_history(self, test_result: TestResult) -> None:
        """Append a test result to the test history."""
        state = self.load()
        if state:
            state.test_history.append(test_result.model_dump())
            self.save(state)

    def add_error(self, error: str) -> None:
        """Append an error to the error list."""
        state = self.load()
        if state:
            state.add_error(error)
            self.save(state)

    def set_status(self, status: SessionStatus) -> None:
        """Set the session status."""
        state = self.load()
        if state:
            state.set_status(status)
            self.save(state)

    def get_state(self) -> Optional[SessionState]:
        """Load and return current session state."""
        return self.load()
