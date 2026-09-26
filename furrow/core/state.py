from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import aiofiles
import structlog
from pydantic import BaseModel, Field

from furrow.config import TaskModel, TestResult


class State(BaseModel):
    """Persistent state tracked across cycles and sessions."""

    goal: str
    cycle: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    test_result: TestResult | None = None
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


class StateManager:
    """Load and save :class:`State` to a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @staticmethod
    def default_path() -> Path:
        return Path.cwd() / ".kilo" / "furrow_state.json"

    async def load(self) -> State | None:
        if not self.path.exists():
            return None
        try:
            async with aiofiles.open(self.path, "r") as f:
                raw = await f.read()
            data = json.loads(raw)
            return State(**data)
        except (json.JSONDecodeError, ValueError) as e:
            structlog.get_logger(__name__).warning(
                "state_load_failed", path=str(self.path), error=str(e)
            )
            return None

    async def save(self, state: State) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        state.touch()
        payload = state.model_dump(mode="json")
        async with aiofiles.open(self.path, "w") as f:
            await f.write(json.dumps(payload, indent=2))
