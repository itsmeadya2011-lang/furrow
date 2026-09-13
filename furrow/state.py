from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel

from furrow.config import Settings, TaskModel, TestResult


DEFAULT_STATE_FILE = Path(".furrow/state.json")


class State(BaseModel):
    goal: str
    cycles: int
    tasks: list[TaskModel]
    last_test_result: TestResult | None
    last_updated: float

    def save(self, path: Path | None = None) -> None:
        target = path or DEFAULT_STATE_FILE
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load(cls, path: Path | None = None) -> State | None:
        target = path or DEFAULT_STATE_FILE
        if not target.exists():
            return None
        with target.open("r") as f:
            data = json.load(f)
        return cls(**data)

    @classmethod
    def create_initial(cls, goal: str, settings: Settings | None = None) -> State:
        return cls(
            goal=goal,
            cycles=0,
            tasks=[],
            last_test_result=None,
            last_updated=time.time(),
        )

    def update_from_orchestrator(self, orchestrator: "Orchestrator") -> None:
        from furrow.core.orchestrator import Orchestrator

        self.goal = orchestrator.goal
        self.cycles = orchestrator.cycles
        self.tasks = orchestrator.tasks
        self.last_test_result = orchestrator.last_test_result
        self.last_updated = time.time()