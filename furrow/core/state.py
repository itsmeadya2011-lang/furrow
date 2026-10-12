from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import structlog

from furrow.config import Plan, TaskModel


logger = structlog.get_logger()


class StateManager:
    """Manages persistent state for the orchestrator."""

    def __init__(self, workspace: Path, state_file: str = ".furrow_state.json") -> None:
        self.workspace = workspace
        self.state_file = workspace / state_file
        self.state: dict[str, Any] = {}
        self.load()

    def load(self) -> dict[str, Any]:
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    self.state = json.load(f)
                logger.info("state_loaded", path=str(self.state_file))
            except Exception as e:
                logger.warning("state_load_failed", error=str(e))
                self.state = {}
        else:
            self.state = {}
        return self.state

    def save(self) -> None:
        """Save state to file."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=2, default=str)
            logger.debug("state_saved", path=str(self.state_file))
        except Exception as e:
            logger.error("state_save_failed", error=str(e))

    def get_tasks(self) -> list[TaskModel]:
        """Get tasks from state."""
        tasks_data = self.state.get("tasks", [])
        return [TaskModel(**t) for t in tasks_data]

    def set_tasks(self, tasks: list[TaskModel]) -> None:
        """Save tasks to state."""
        self.state["tasks"] = [t.model_dump() for t in tasks]
        self.save()

    def get_plan(self) -> Plan | None:
        """Get plan from state."""
        plan_data = self.state.get("plan")
        if plan_data:
            return Plan(**plan_data)
        return None

    def set_plan(self, plan: Plan) -> None:
        """Save plan to state."""
        self.state["plan"] = plan.model_dump()
        self.save()

    def get_goal(self) -> str:
        """Get goal from state."""
        return self.state.get("goal", "")

    def set_goal(self, goal: str) -> None:
        """Save goal to state."""
        self.state["goal"] = goal
        self.save()

    def get_cycles(self) -> int:
        """Get cycle count from state."""
        return self.state.get("cycles", 0)

    def set_cycles(self, cycles: int) -> None:
        """Save cycle count to state."""
        self.state["cycles"] = cycles
        self.save()

    def clear(self) -> None:
        """Clear state file."""
        if self.state_file.exists():
            self.state_file.unlink()
        self.state = {}
        logger.info("state_cleared")