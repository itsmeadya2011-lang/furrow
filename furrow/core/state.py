"""Persistent state management for the Furrow orchestrator.

State is stored as a JSON file (``furrow-state.json``) in the workspace
directory so that task/goal progress survives across sessions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from furrow.config import TaskModel

STATE_FILENAME = "furrow-state.json"


@dataclass
class CycleRecord:
    """Record of a single development loop cycle."""

    cycle: int
    plan: dict | None = None
    tasks: list[TaskModel] = field(default_factory=list)
    test_result: dict | None = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "cycle": self.cycle,
            "plan": self.plan,
            "tasks": [t.model_dump() for t in self.tasks],
            "test_result": self.test_result,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentState:
    """Full persistent state for an Orchestrator session."""

    goal: str
    cycles: list[CycleRecord] = field(default_factory=list)
    completed_tasks: set[str] = field(default_factory=set)
    failed_tasks: set[str] = field(default_factory=set)
    running: bool = False

    @classmethod
    def load(cls, workspace: str | Path) -> AgentState:
        """Load state from the workspace. Returns a fresh state if not found."""
        path = Path(workspace) / STATE_FILENAME
        if path.exists():
            try:
                data = json.loads(path.read_text())
                return cls._from_dict(data)
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        return cls(goal="")

    @classmethod
    def _from_dict(cls, data: dict) -> AgentState:
        state = AgentState(goal=data.get("goal", ""))
        state.running = data.get("running", False)
        state.completed_tasks = set(data.get("completed_tasks", []))
        state.failed_tasks = set(data.get("failed_tasks", []))
        for cyc in data.get("cycles", []):
            tasks = [TaskModel(**t) for t in cyc.get("tasks", [])]
            state.cycles.append(
                CycleRecord(
                    cycle=cyc["cycle"],
                    plan=cyc.get("plan"),
                    tasks=tasks,
                    test_result=cyc.get("test_result"),
                    timestamp=cyc.get("timestamp", ""),
                )
            )
        return state

    def save(self, workspace: str | Path) -> None:
        """Persist state to the workspace directory."""
        path = Path(workspace) / STATE_FILENAME
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "cycles": [c.to_dict() for c in self.cycles],
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "running": self.running,
        }

    def add_cycle(self, record: CycleRecord) -> None:
        self.cycles.append(record)
        # Update completed/failed sets from task statuses
        for task in record.tasks:
            if task.status == "completed":
                self.completed_tasks.add(task.id)
                self.failed_tasks.discard(task.id)
            elif task.status == "failed":
                self.failed_tasks.add(task.id)
                self.completed_tasks.discard(task.id)
            elif task.status == "pending":
                self.completed_tasks.discard(task.id)
                self.failed_tasks.discard(task.id)

    def mark_task_completed(self, task_id: str) -> None:
        self.completed_tasks.add(task_id)
        self.failed_tasks.discard(task_id)

    def mark_task_failed(self, task_id: str) -> None:
        self.failed_tasks.add(task_id)
        self.completed_tasks.discard(task_id)

    @property
    def all_tasks(self) -> list[TaskModel]:
        """Return all tasks from all cycles combined."""
        all_tasks: list[TaskModel] = []
        for cycle in self.cycles:
            all_tasks.extend(cycle.tasks)
        return all_tasks

    @property
    def pending_tasks(self) -> list[TaskModel]:
        """Return tasks that are still pending or failed (need work)."""
        return [t for t in self.all_tasks if t.status in ("pending", "failed")]

    @property
    def completed_count(self) -> int:
        return len(self.completed_tasks)

    @property
    def failed_count(self) -> int:
        return len(self.failed_tasks)
