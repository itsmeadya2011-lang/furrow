from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, UTC

import aiofiles

from furrow.config import Plan, TaskModel, TestResult


class OrchestratorState:
    """Manages persistent state for the orchestrator across sessions."""

    def __init__(self, state_file: Path | str | None = None) -> None:
        self.state_file = Path(state_file) if state_file else Path(".furrow_state.json")
        self.data: dict[str, Any] = {
            "goal": "",
            "cycles": 0,
            "test_passed": False,
            "tasks": [],
            "history": [],
            "last_updated": None,
        }

    async def save(self) -> None:
        """Save current state to file."""
        self.data["last_updated"] = datetime.now(UTC).isoformat()
        async with aiofiles.open(self.state_file, "w") as f:
            await f.write(json.dumps(self.data, indent=2))

    async def load(self) -> dict[str, Any]:
        """Load state from file."""
        if not self.state_file.exists():
            return self.data
        async with aiofiles.open(self.state_file, "r") as f:
            content = await f.read()
            self.data = json.loads(content)
        return self.data

    def update_from_plan(self, plan: Plan) -> None:
        """Update state from a plan."""
        self.data["tasks"] = [
            {
                "id": task.id,
                "description": task.description,
                "files": task.files,
                "status": task.status,
                "result": task.result,
            }
            for task in plan.tasks
        ]

    def add_history_entry(self, cycle: int, test_result: TestResult) -> None:
        """Add a cycle entry to history."""
        self.data["history"].append({
            "cycle": cycle,
            "passed": test_result.passed,
            "summary": test_result.summary,
            "timestamp": datetime.now(UTC).isoformat(),
        })

    @property
    def goal(self) -> str:
        return self.data["goal"]

    @goal.setter
    def goal(self, value: str) -> None:
        self.data["goal"] = value

    @property
    def cycles(self) -> int:
        return self.data["cycles"]

    @cycles.setter
    def cycles(self, value: int) -> None:
        self.data["cycles"] = value

    @property
    def test_passed(self) -> bool:
        return self.data["test_passed"]

    @test_passed.setter
    def test_passed(self, value: bool) -> None:
        self.data["test_passed"] = value

    def clear(self) -> None:
        """Clear all state."""
        self.data = {
            "goal": "",
            "cycles": 0,
            "test_passed": False,
            "tasks": [],
            "history": [],
            "last_updated": None,
        }
