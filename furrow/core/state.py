"""Persistent state management for Furrow development sessions.

Tracks goals, tasks, and progress across cycles and sessions. The state is
serialised to a JSON file (default: ``.furrow/state.json``) so that a
stopped session can be resumed later.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from furrow.config import Plan, TaskModel


class SessionStatus(str, Enum):
    """Lifecycle status of a Furrow session."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SessionState:
    """Runtime state for a single Furrow session."""

    goal: str
    original_goal: str
    status: SessionStatus = SessionStatus.ACTIVE
    cycle: int = 0
    max_cycles: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    tasks: list[TaskModel] = field(default_factory=list)
    plan_history: list[Plan] = field(default_factory=list)
    test_history: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "original_goal": self.original_goal,
            "status": self.status.value,
            "cycle": self.cycle,
            "max_cycles": self.max_cycles,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "tasks": [t.model_dump() for t in self.tasks],
            "plan_history": [p.model_dump() for p in self.plan_history],
            "test_history": self.test_history,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SessionState:
        return cls(
            goal=data["goal"],
            original_goal=data.get("original_goal", data["goal"]),
            status=SessionStatus(data.get("status", "active")),
            cycle=data.get("cycle", 0),
            max_cycles=data.get("max_cycles", 0),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            tasks=[TaskModel(**t) for t in data.get("tasks", [])],
            plan_history=[Plan(**p) for p in data.get("plan_history", [])],
            test_history=data.get("test_history", []),
            errors=data.get("errors", []),
        )


class StateManager:
    """Manages persistent state for Furrow sessions.

    The state is stored as a JSON file so that sessions survive restarts.
    Each orchestrator session corresponds to one state file.
    """

    DEFAULT_STATE_FILE = ".furrow/state.json"

    def __init__(self, state_file: str | Path | None = None) -> None:
        self.state_file = Path(state_file) if state_file else Path(self.DEFAULT_STATE_FILE)
        self._state: Optional[SessionState] = None

    @property
    def state(self) -> SessionState:
        """Return the current session state, loading from disk if needed.

        Raises:
            RuntimeError: If state has not been initialized or loaded.
        """
        if self._state is None:
            self._load()
        if self._state is None:
            raise RuntimeError(
                "State has not been initialized. Call initialize() or load() first."
            )
        return self._state

    def initialize(self, goal: str, max_cycles: int = 0) -> SessionState:
        """Create a fresh session state for a new goal."""
        self._state = SessionState(
            goal=goal,
            original_goal=goal,
            max_cycles=max_cycles,
        )
        self.save()
        return self._state

    def load(self) -> Optional[SessionState]:
        """Load state from disk. Returns None if no state file exists."""
        return self._load()

    def _load(self) -> Optional[SessionState]:
        if not self.state_file.exists():
            return None
        try:
            data = json.loads(self.state_file.read_text())
            self._state = SessionState.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            # Corrupt state file — reset
            self._state = None
        return self._state

    def save(self) -> None:
        """Persist current state to disk."""
        if self._state is None:
            return
        self._state.updated_at = time.time()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(self._state.to_dict(), indent=2))

    # -- Convenience accessors ------------------------------------------------

    def update_tasks(self, tasks: list[TaskModel]) -> None:
        """Replace the task list with updated statuses from a new plan."""
        existing = {t.id: t for t in self.state.tasks}
        for task in tasks:
            if task.id in existing:
                # Preserve completed status if not in the new plan
                old = existing[task.id]
                if old.status == "completed" and task.status != "failed":
                    task.status = old.status
                    task.result = old.result
        self.state.tasks = tasks
        self.save()

    def mark_task_completed(self, task_id: str, result: str) -> None:
        for task in self.state.tasks:
            if task.id == task_id:
                task.status = "completed"
                task.result = result
                break
        self.save()

    def mark_task_failed(self, task_id: str, error: str) -> None:
        for task in self.state.tasks:
            if task.id == task_id:
                task.status = "failed"
                task.result = error
                break
        self.save()

    def add_error(self, error: str) -> None:
        self.state.errors.append(error)
        self.save()

    def increment_cycle(self) -> int:
        self.state.cycle += 1
        self.save()
        return self.state.cycle

    def all_tasks_done(self) -> bool:
        tasks = self.state.tasks
        if not tasks:
            return False
        return all(t.status in ("completed", "failed") for t in tasks)

    def has_failures(self) -> bool:
        return any(t.status == "failed" for t in self.state.tasks)

    def completed_count(self) -> int:
        return sum(1 for t in self.state.tasks if t.status == "completed")

    def failed_count(self) -> int:
        return sum(1 for t in self.state.tasks if t.status == "failed")

    def is_cycle_limit_reached(self) -> bool:
        if self.state.max_cycles <= 0:
            return False
        return self.state.cycle >= self.state.max_cycles

    def set_goal(self, goal: str) -> None:
        """Update the current goal (e.g. when tests fail and need fixing)."""
        self.state.goal = goal
        self.save()

    def complete(self) -> None:
        self.state.status = SessionStatus.COMPLETED
        self.save()

    def fail(self, reason: str) -> None:
        self.state.status = SessionStatus.FAILED
        self.state.errors.append(reason)
        self.save()

    def get_task(self, task_id: str) -> Optional[TaskModel]:
        for task in self.state.tasks:
            if task.id == task_id:
                return task
        return None
