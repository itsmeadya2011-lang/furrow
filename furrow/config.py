from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

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
    result: Optional[str] = None
    cycle: int = 0


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
    plan: Plan
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class FurrowState(BaseModel):
    goal: str
    cycles: list[CycleState] = Field(default_factory=list)
    current_cycle: int = 0
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def add_cycle(self, cycle: int, goal: str, plan: Plan) -> None:
        self.cycles.append(CycleState(cycle=cycle, goal=goal, plan=plan))
        self.current_cycle = cycle
        self.updated_at = datetime.now().isoformat()

    def get_all_tasks(self) -> list[TaskModel]:
        tasks = []
        for cycle_state in self.cycles:
            for task in cycle_state.plan.tasks:
                task.cycle = cycle_state.cycle
                tasks.append(task)
        return tasks

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "FurrowState":
        if path.exists():
            return cls.model_validate_json(path.read_text())
        return cls(goal="")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    state_file: Path = Field(default_factory=lambda: Path.cwd() / ".furrow_state.json")


settings = Settings()
