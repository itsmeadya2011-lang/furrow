from __future__ import annotations

import time
from pathlib import Path

import aiofiles
from pydantic import BaseModel, Field

from furrow.config import TaskModel


class State(BaseModel):
    goal: str
    tasks: list[TaskModel] = Field(default_factory=list)
    cycles: int = 0
    completed: int = 0
    failed: int = 0
    last_updated: float = 0.0
    status: str = "running"


async def save_state(state: State, path: Path) -> None:
    state.last_updated = time.time()
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w") as f:
        await f.write(state.model_dump_json(indent=2))


async def load_state(path: Path) -> State | None:
    try:
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
        return State.model_validate_json(content)
    except (FileNotFoundError, ValueError):
        return None
