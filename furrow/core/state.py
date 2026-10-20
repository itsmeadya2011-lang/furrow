from __future__ import annotations

import contextlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from furrow.config import TaskModel

if TYPE_CHECKING:
    from furrow.core.orchestrator import Orchestrator


class RunState(BaseModel):
    goal: str
    original_goal: str
    cycles: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    history: list[dict[str, Any]] = Field(default_factory=list)
    updated_at: str = ""
    version: int = 1


def state_from_orchestrator(orchestrator: Orchestrator) -> RunState:
    return RunState(
        goal=orchestrator.goal,
        original_goal=orchestrator.original_goal,
        cycles=orchestrator.cycles,
        tasks=list(orchestrator.tasks),
        history=[dict(h) for h in orchestrator.history],
        updated_at=datetime.now(timezone.utc).isoformat(),
        version=1,
    )


def save_state(state: RunState, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = state.model_dump(mode="json")
    payload = json.dumps(data, indent=2, sort_keys=False)

    fd, tmp_path = tempfile.mkstemp(prefix=".state_", suffix=".tmp", dir=str(path.parent))
    try:
        os.write(fd, payload.encode("utf-8"))
        os.close(fd)
        fd = -1
        os.replace(tmp_path, path)
    finally:
        if fd != -1:
            with contextlib.suppress(OSError):
                os.close(fd)
        if tmp_path and os.path.exists(tmp_path):
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    return path


def load_state(path: str | Path) -> RunState:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No saved state found at {path}. Start a new run with `furrow start \"<goal>\"`."
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Saved state at {path} is corrupt: {exc}") from exc

    try:
        return RunState.model_validate(data)
    except Exception as exc:
        raise ValueError(f"Saved state at {path} is invalid: {exc}") from exc


def restore_orchestrator(orchestrator: Orchestrator, state: RunState) -> Orchestrator:
    orchestrator.goal = state.goal
    orchestrator.original_goal = state.original_goal
    orchestrator.cycles = state.cycles
    orchestrator.tasks = list(state.tasks)
    orchestrator.history = [dict(h) for h in state.history]
    return orchestrator
