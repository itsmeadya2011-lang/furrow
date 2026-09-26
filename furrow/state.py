from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel

from furrow.config import Settings, TaskModel, TestResult, settings


class FurrowState(BaseModel):
    goal: str
    cycles: int
    all_tasks: list[TaskModel]
    last_test_result: TestResult | None = None
    updated_at: float = 0.0

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads(self.model_dump_json())
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> FurrowState | None:
        if not path.exists():
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return cls.model_validate(data)
        except (json.JSONDecodeError, ValueError, OSError):
            return None
