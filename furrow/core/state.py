from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog
from pydantic import BaseModel, Field, PrivateAttr, ValidationError

from furrow.config import Settings, TaskModel, TestResult
from furrow.config import settings as default_settings

log = structlog.get_logger(__name__)

STATE_FILE_NAME = ".furrow-state.json"


def _resolve_state_path(
    path: str | Path | None,
    settings: Settings,
) -> Path:
    if path is not None:
        return Path(path)
    return Path(settings.workspace) / STATE_FILE_NAME


def _read_state_payload(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text()
    except FileNotFoundError:
        return None
    except OSError as e:
        log.warning("Failed to read state file", path=str(path), error=str(e))
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        log.warning("State file is not valid JSON", path=str(path), error=str(e))
        return None


class FurrowState(BaseModel):
    goal: str = ""
    cycles: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    test_results: list[TestResult] = Field(default_factory=list)
    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    status: str = "running"

    _path: Path = PrivateAttr(default=None)

    def __init__(
        self,
        path: str | Path | None = None,
        settings: Settings | None = None,
        **data: Any,
    ) -> None:
        resolved_settings = settings or default_settings
        resolved_path = _resolve_state_path(path, resolved_settings)

        if not data:
            payload = _read_state_payload(resolved_path)
            if payload is not None:
                data = payload

        try:
            super().__init__(**data)
        except ValidationError as e:
            log.warning(
                "State file validation failed, using defaults",
                path=str(resolved_path),
                error=str(e),
            )
            super().__init__()

        self._path = resolved_path

    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        settings: Settings | None = None,
    ) -> FurrowState | None:
        resolved_settings = settings or default_settings
        state_path = _resolve_state_path(path, resolved_settings)

        if not state_path.exists():
            return None

        payload = _read_state_payload(state_path)
        if payload is None:
            return None

        try:
            state = cls.model_validate(payload)
        except ValidationError as e:
            log.warning(
                "State file validation failed",
                path=str(state_path),
                error=str(e),
            )
            return None

        state._path = state_path
        return state

    def save(self) -> None:
        self.last_updated = datetime.now(timezone.utc)
        try:
            payload = self.model_dump(mode="json")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(payload, indent=2))
        except OSError as e:
            log.warning(
                "Failed to save state file",
                path=str(self._path),
                error=str(e),
            )

    def add_task(self, task: TaskModel) -> None:
        self.tasks.append(task)
        self.last_updated = datetime.now(timezone.utc)

    def update_task(
        self,
        task_id: str,
        status: str,
        result: str | None = None,
    ) -> None:
        for task in self.tasks:
            if task.id == task_id:
                task.status = status
                if result is not None:
                    task.result = result
                self.last_updated = datetime.now(timezone.utc)
                return
        log.warning("Task not found for update", task_id=task_id)

    def add_test_result(self, result: TestResult) -> None:
        self.test_results.append(result)
        self.last_updated = datetime.now(timezone.utc)

    def set_status(self, status: str) -> None:
        self.status = status
        self.last_updated = datetime.now(timezone.utc)

    def set_goal(self, goal: str) -> None:
        self.goal = goal
        self.last_updated = datetime.now(timezone.utc)

    @property
    def path(self) -> Path:
        return self._path
