from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import structlog
from pydantic import BaseModel, Field

from furrow.config import TaskModel

if TYPE_CHECKING:
    from furrow.config import Plan

logger = structlog.get_logger(__name__)


class TaskState(BaseModel):
    """Represents the state of a single task in a run."""

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None


class RunState(BaseModel):
    """Represents the complete state of a Furrow run."""

    goal: str
    cycles: int = 0
    tasks: list[TaskState] = Field(default_factory=list)
    history: list[str] = Field(default_factory=list)
    created_at: str
    updated_at: str

    @classmethod
    def new(cls, goal: str, tasks: list[TaskState] | None = None) -> RunState:
        """Create a new RunState with current timestamps."""
        now = datetime.now().isoformat()
        return cls(goal=goal, created_at=now, updated_at=now, tasks=tasks or [])


class StateManager:
    """Manages persistent state for Furrow runs."""

    DEFAULT_STATE_FILE: ClassVar[str] = ".furrow-state.json"

    def __init__(self, path: Path | str | None = None) -> None:
        """Initialize the StateManager.

        Args:
            path: Path to the state file. Defaults to .furrow-state.json in cwd.
        """
        if path is None:
            self.path = Path.cwd() / self.DEFAULT_STATE_FILE
        elif isinstance(path, str):
            self.path = Path(path)
        else:
            self.path = path
        logger.debug("StateManager initialized", path=str(self.path))

    def load(self) -> RunState | None:
        """Load state from the JSON file.

        Returns:
            RunState if file exists and parses successfully, None otherwise.
        """
        if not self.path.exists():
            logger.debug("State file not found", path=str(self.path))
            return None

        try:
            content = self.path.read_text(encoding="utf-8")
            data = json.loads(content)
            state = RunState.model_validate(data)
            logger.info("State loaded", path=str(self.path), cycles=state.cycles, tasks=len(state.tasks))
            return state
        except (json.JSONDecodeError, OSError, ValueError) as e:
            logger.warning("Failed to load state", path=str(self.path), error=str(e))
            return None

    def save(self, state: RunState) -> None:
        """Save state to the JSON file.

        Args:
            state: The RunState to persist.
        """
        state.updated_at = datetime.now().isoformat()
        try:
            self.path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
            logger.info("State saved", path=str(self.path), cycles=state.cycles, tasks=len(state.tasks))
        except OSError as e:
            logger.error("Failed to save state", path=str(self.path), error=str(e))
            raise

    @staticmethod
    def from_plan(plan: Plan, goal: str) -> RunState:
        """Create a RunState from a Plan and goal.

        Args:
            plan: The Plan containing tasks.
            goal: The goal description for the run.

        Returns:
            A new RunState initialized with tasks from the plan.
        """
        tasks = [
            TaskState(
                id=task.id,
                description=task.description,
                files=task.files,
                status=task.status,
                result=task.result,
            )
            for task in plan.tasks
        ]
        return RunState.new(goal=goal, tasks=tasks)