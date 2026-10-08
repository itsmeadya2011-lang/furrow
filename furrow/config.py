from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None
    cycle: int = 0
    attempts: int = 0


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class CycleState(BaseModel):
    cycle: int
    goal: str
    tasks: list[TaskModel]
    test_result: TestResult | None = None
    timestamp: str = ""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    state_file: Path = Field(default_factory=lambda: Path.cwd() / ".furrow_state.json")


class StateManager:
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.cycles: list[CycleState] = []
        self.current_goal: str = ""
        self.all_tasks: list[TaskModel] = []
        self._load()

    def _load(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.cycles = [CycleState(**c) for c in data.get("cycles", [])]
                self.current_goal = data.get("current_goal", "")
                self.all_tasks = [TaskModel(**t) for t in data.get("all_tasks", [])]
            except Exception:  # noqa: BLE001, S110
                pass

    def save(self) -> None:
        data = {
            "current_goal": self.current_goal,
            "all_tasks": [t.model_dump() for t in self.all_tasks],
            "cycles": [c.model_dump() for c in self.cycles],
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    def add_cycle(self, cycle: int, goal: str, tasks: list[TaskModel], test_result: TestResult | None = None) -> None:
        self.cycles.append(CycleState(
            cycle=cycle,
            goal=goal,
            tasks=tasks,
            test_result=test_result,
            timestamp=datetime.now(timezone.utc).isoformat(),
        ))
        self.all_tasks = tasks
        self.current_goal = goal
        self.save()

    def get_pending_tasks(self) -> list[TaskModel]:
        return [t for t in self.all_tasks if t.status in ("pending", "in_progress")]

    def get_completed_tasks(self) -> list[TaskModel]:
        return [t for t in self.all_tasks if t.status == "completed"]

    def get_failed_tasks(self) -> list[TaskModel]:
        return [t for t in self.all_tasks if t.status == "failed"]

    def is_done(self) -> bool:
        pending = self.get_pending_tasks()
        failed = self.get_failed_tasks()
        if failed:
            return False
        return not pending and bool(self.all_tasks)


settings = Settings()
