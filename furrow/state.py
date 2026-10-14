from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Literal

import structlog
from pydantic import BaseModel, Field, ValidationError

from furrow.config import TaskModel, TestResult, settings

logger = structlog.get_logger(__name__)


class FurrowState(BaseModel):
    goal: str
    cycles: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    last_test_result: TestResult | None = None
    status: Literal["pending", "running", "complete", "stopped"] = "pending"
    updated_at: datetime = Field(default_factory=datetime.now)


class StateStore:
    def __init__(
        self,
        path: Path | None = None,
        workspace: Path | None = None,
    ) -> None:
        if path is not None:
            self._path = path
        else:
            ws = workspace if workspace is not None else settings.workspace
            self._path = ws / ".furrow" / "state.json"

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> FurrowState | None:
        p = self.path
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("state.load.failed", path=str(p), error=str(exc))
            return None
        try:
            return FurrowState(**data)
        except (ValidationError, ValueError, TypeError) as exc:
            logger.warning("state.load.invalid", path=str(p), error=str(exc))
            return None

    def save(self, state: FurrowState) -> None:
        p = self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        data = state.model_dump_json(indent=2)
        p.write_text(data, encoding="utf-8")
        logger.debug("state.save.complete", path=str(p))

    def delete(self) -> None:
        p = self.path
        if p.exists():
            p.unlink()
            logger.debug("state.delete.complete", path=str(p))
