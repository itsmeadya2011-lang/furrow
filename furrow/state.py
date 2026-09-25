from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from furrow.config import Plan, Settings, TaskModel, TestResult


class StateManager:
    def __init__(self, settings: Settings = Settings()) -> None:
        self.settings = settings
        self.state_file = settings.workspace / ".furrow_state.json"
        self._state: dict[str, Any] = {
            "goal": "",
            "original_goal": "",
            "tasks": [],
            "cycles": 0,
            "max_cycles": settings.max_cycles,
            "history": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    loaded = json.load(f)
                    self._state.update(loaded)
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self) -> None:
        self._state["updated_at"] = datetime.now().isoformat()
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    @property
    def goal(self) -> str:
        return self._state.get("goal", "")

    @goal.setter
    def goal(self, value: str) -> None:
        self._state["goal"] = value
        self._save()

    @property
    def original_goal(self) -> str:
        return self._state.get("original_goal", "")

    @original_goal.setter
    def original_goal(self, value: str) -> None:
        self._state["original_goal"] = value
        self._save()

    @property
    def tasks(self) -> list[dict[str, Any]]:
        return self._state.get("tasks", [])

    @tasks.setter
    def tasks(self, value: list[dict[str, Any]]) -> None:
        self._state["tasks"] = value
        self._save()

    @property
    def cycles(self) -> int:
        return self._state.get("cycles", 0)

    @cycles.setter
    def cycles(self, value: int) -> None:
        self._state["cycles"] = value
        self._save()

    @property
    def max_cycles(self) -> int:
        return self._state.get("max_cycles", 0)

    @max_cycles.setter
    def max_cycles(self, value: int) -> None:
        self._state["max_cycles"] = value
        self._save()

    @property
    def history(self) -> list[dict[str, Any]]:
        return self._state.get("history", [])

    def add_history_entry(self, cycle: int, plan: Plan, test_result: TestResult) -> None:
        entry = {
            "cycle": cycle,
            "timestamp": datetime.now().isoformat(),
            "plan": plan.model_dump(),
            "test_result": test_result.model_dump(),
        }
        self._state.setdefault("history", []).append(entry)
        self._save()

    def get_pending_tasks(self) -> list[TaskModel]:
        return [TaskModel(**t) for t in self.tasks if t.get("status") == "pending"]

    def get_completed_tasks(self) -> list[TaskModel]:
        return [TaskModel(**t) for t in self.tasks if t.get("status") == "completed"]

    def get_failed_tasks(self) -> list[TaskModel]:
        return [TaskModel(**t) for t in self.tasks if t.get("status") == "failed"]

    def update_task(self, task_id: str, **updates: Any) -> None:
        for task in self.tasks:
            if task["id"] == task_id:
                task.update(updates)
                break
        self._save()

    def is_done(self) -> bool:
        if self.max_cycles > 0 and self.cycles >= self.max_cycles:
            return True
        all_tasks = self.tasks
        if not all_tasks:
            return False
        completed = sum(1 for t in all_tasks if t.get("status") == "completed")
        failed = sum(1 for t in all_tasks if t.get("status") == "failed")
        if failed > 0:
            return False
        return completed >= len(all_tasks)

    def clear(self) -> None:
        self._state = {
            "goal": "",
            "original_goal": "",
            "tasks": [],
            "cycles": 0,
            "max_cycles": self.settings.max_cycles,
            "history": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._save()
        if self.state_file.exists():
            self.state_file.unlink()

    def to_dict(self) -> dict[str, Any]:
        return self._state.copy()