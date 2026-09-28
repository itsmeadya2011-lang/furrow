from __future__ import annotations

import json
from pathlib import Path

from furrow.config import TaskModel, TestResult


class State:
    """Persistent state for the Furrow orchestrator.

    Saves the current goal, cycle count, task statuses, and the most recent
    test result to a JSON file so an interrupted run can be resumed.
    """

    def __init__(self, path: Path = Path(".furrow/state.json")) -> None:
        self.path = path
        self.goal: str = ""
        self.cycles: int = 0
        self.tasks: list[TaskModel] = []
        self.last_test_result: TestResult | None = None

    def load(self) -> None:
        """Load state from disk. No-op if the file is missing or corrupt."""
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError):
            return
        self.goal = data.get("goal", "")
        self.cycles = data.get("cycles", 0)
        self.tasks = [TaskModel(**t) for t in data.get("tasks", [])]
        last = data.get("last_test_result")
        if last:
            try:
                self.last_test_result = TestResult(**last)
            except (ValueError, TypeError):
                self.last_test_result = None

    def save(self) -> None:
        """Persist state to disk as JSON."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "goal": self.goal,
            "cycles": self.cycles,
            "tasks": [t.model_dump() for t in self.tasks],
            "last_test_result": (
                self.last_test_result.model_dump() if self.last_test_result else None
            ),
        }
        self.path.write_text(json.dumps(payload, indent=2))

    def update(
        self,
        *,
        goal: str | None = None,
        cycles: int | None = None,
        tasks: list[TaskModel] | None = None,
        last_test_result: TestResult | None = None,
    ) -> None:
        """Update one or more fields and immediately persist to disk."""
        if goal is not None:
            self.goal = goal
        if cycles is not None:
            self.cycles = cycles
        if tasks is not None:
            self.tasks = tasks
        if last_test_result is not None:
            self.last_test_result = last_test_result
        self.save()
