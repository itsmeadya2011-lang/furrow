from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from furrow.config import Plan, TaskModel, settings


@dataclass
class CycleRecord:
    cycle: int
    goal: str
    plan: Plan
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    test_result: dict[str, Any] | None = None
    completed_tasks: list[str] = field(default_factory=list)
    failed_tasks: list[str] = field(default_factory=list)


@dataclass
class FurrowState:
    goal: str
    cycles: list[CycleRecord] = field(default_factory=list)
    current_cycle: int = 0
    total_tasks_completed: int = 0
    total_tasks_failed: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def add_cycle(self, record: CycleRecord) -> None:
        self.cycles.append(record)
        self.current_cycle = record.cycle
        self.updated_at = datetime.now().isoformat()
        completed = sum(1 for t in record.plan.tasks if t.id in record.completed_tasks)
        failed = sum(1 for t in record.plan.tasks if t.id in record.failed_tasks)
        self.total_tasks_completed += completed
        self.total_tasks_failed += failed

    def get_pending_tasks(self) -> list[TaskModel]:
        """Get tasks that are not yet completed from the latest cycle."""
        if not self.cycles:
            return []
        latest = self.cycles[-1]
        completed_ids = set(latest.completed_tasks)
        return [t for t in latest.plan.tasks if t.id not in completed_ids and t.id not in latest.failed_tasks]

    def get_progress_summary(self) -> dict[str, Any]:
        """Get a summary of overall progress."""
        total_planned = sum(len(c.plan.tasks) for c in self.cycles)
        total_completed = self.total_tasks_completed
        total_failed = self.total_tasks_failed
        return {
            "goal": self.goal,
            "current_cycle": self.current_cycle,
            "total_cycles": len(self.cycles),
            "total_tasks_planned": total_planned,
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "completion_rate": total_completed / total_planned if total_planned > 0 else 0,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "cycles": [
                {
                    "cycle": c.cycle,
                    "goal": c.goal,
                    "plan": c.plan.model_dump(),
                    "timestamp": c.timestamp,
                    "test_result": c.test_result,
                    "completed_tasks": c.completed_tasks,
                    "failed_tasks": c.failed_tasks,
                }
                for c in self.cycles
            ],
            "current_cycle": self.current_cycle,
            "total_tasks_completed": self.total_tasks_completed,
            "total_tasks_failed": self.total_tasks_failed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FurrowState:
        state = cls(goal=data["goal"])
        state.current_cycle = data.get("current_cycle", 0)
        state.total_tasks_completed = data.get("total_tasks_completed", 0)
        state.total_tasks_failed = data.get("total_tasks_failed", 0)
        state.created_at = data.get("created_at", state.created_at)
        state.updated_at = data.get("updated_at", state.updated_at)
        for c in data.get("cycles", []):
            plan = Plan(**c["plan"])
            record = CycleRecord(
                cycle=c["cycle"],
                goal=c["goal"],
                plan=plan,
                timestamp=c.get("timestamp", datetime.now().isoformat()),
                test_result=c.get("test_result"),
                completed_tasks=c.get("completed_tasks", []),
                failed_tasks=c.get("failed_tasks", []),
            )
            state.cycles.append(record)
        return state


class StateManager:
    def __init__(self, workspace: Path | None = None) -> None:
        self.workspace = workspace or settings.workspace
        self.state_file = self.workspace / ".furrow_state.json"
        self._state: FurrowState | None = None

    def load(self) -> FurrowState | None:
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r") as f:
                data = json.load(f)
            self._state = FurrowState.from_dict(data)
            return self._state
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: Failed to load state: {e}")
            return None

    def save(self, state: FurrowState) -> None:
        self._state = state
        try:
            with open(self.state_file, "w") as f:
                json.dump(state.to_dict(), f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save state: {e}")

    def get_state(self) -> FurrowState | None:
        return self._state

    def clear(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()
        self._state = None

    def create_new(self, goal: str) -> FurrowState:
        state = FurrowState(goal=goal)
        self.save(state)
        return state