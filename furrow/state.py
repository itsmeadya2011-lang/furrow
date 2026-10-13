from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field

from furrow.config import TaskModel

logger = structlog.get_logger()


class CycleRecord(BaseModel):
    cycle: int
    timestamp: str
    goal: str
    plan_rationale: str
    tasks: list[dict[str, Any]]
    test_result: dict[str, Any] | None = None


class FurrowState(BaseModel):
    goal: str = ""
    cycles: list[CycleRecord] = Field(default_factory=list)
    current_cycle: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @property
    def completed_tasks(self) -> list[dict[str, Any]]:
        if not self.cycles:
            return []
        latest = self.cycles[-1]
        return [t for t in latest.tasks if t.get("status") == "completed"]

    @property
    def failed_tasks(self) -> list[dict[str, Any]]:
        if not self.cycles:
            return []
        latest = self.cycles[-1]
        return [t for t in latest.tasks if t.get("status") == "failed"]

    @property
    def pending_tasks(self) -> list[dict[str, Any]]:
        if not self.cycles:
            return []
        latest = self.cycles[-1]
        return [t for t in latest.tasks if t.get("status") == "pending"]

    @property
    def is_complete(self) -> bool:
        if not self.cycles:
            return False
        latest = self.cycles[-1]
        if not latest.tasks:
            return True
        return all(t.get("status") == "completed" for t in latest.tasks)


class StateManager:
    def __init__(self, workspace: Path | str = ".", state_file: str = ".furrow_state.json") -> None:
        self.workspace = Path(workspace)
        self.state_file = self.workspace / state_file
        self._state: FurrowState | None = None

    def load(self) -> FurrowState:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self._state = FurrowState(**data)
                logger.info("state_loaded", file=str(self.state_file), cycles=len(self._state.cycles))
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning("state_load_failed", error=str(e), file=str(self.state_file))
                self._state = FurrowState()
        else:
            self._state = FurrowState()
            logger.info("state_created", file=str(self.state_file))
        return self._state

    def save(self) -> None:
        if self._state is None:
            return
        self._state.updated_at = datetime.now().isoformat()
        self.state_file.write_text(self._state.model_dump_json(indent=2))
        logger.debug("state_saved", file=str(self.state_file))

    def start_cycle(self, goal: str, plan_rationale: str, tasks: list[TaskModel]) -> CycleRecord:
        if self._state is None:
            self.load()
        cycle_num = self._state.current_cycle + 1
        self._state.current_cycle = cycle_num
        self._state.goal = goal

        cycle = CycleRecord(
            cycle=cycle_num,
            timestamp=datetime.now().isoformat(),
            goal=goal,
            plan_rationale=plan_rationale,
            tasks=[t.model_dump() for t in tasks],
        )
        self._state.cycles.append(cycle)
        self.save()
        return cycle

    def update_task_result(self, task_id: str, status: str, result: str | None = None) -> None:
        if self._state is None or not self._state.cycles:
            return
        latest = self._state.cycles[-1]
        for task in latest.tasks:
            if task.get("id") == task_id:
                task["status"] = status
                if result is not None:
                    task["result"] = result
                break
        self.save()

    def set_test_result(self, passed: bool, summary: str, failures: list[str]) -> None:
        if self._state is None or not self._state.cycles:
            return
        latest = self._state.cycles[-1]
        latest.test_result = {
            "passed": passed,
            "summary": summary,
            "failures": failures,
        }
        self.save()

    def get_state(self) -> FurrowState:
        if self._state is None:
            return self.load()
        return self._state

    def reset(self) -> None:
        self._state = FurrowState()
        if self.state_file.exists():
            self.state_file.unlink()
        logger.info("state_reset")


def get_state_manager(workspace: Path | str = ".", state_file: str = ".furrow_state.json") -> StateManager:
    return StateManager(workspace, state_file)