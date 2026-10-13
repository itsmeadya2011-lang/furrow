from __future__ import annotations

import json
from datetime import datetime
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
    plan: Plan | None = None
    test_result: TestResult | None = None
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class State(BaseModel):
    goal: str
    cycles: list[CycleState] = Field(default_factory=list)
    current_cycle: int = 0
    completed_tasks: list[TaskModel] = Field(default_factory=list)
    failed_tasks: list[TaskModel] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def load(cls, path: Path) -> State:
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return cls(goal="")

    def save(self, path: Path) -> None:
        self.updated_at = datetime.now().isoformat()
        path.write_text(json.dumps(self.model_dump(), indent=2))

    def add_cycle(self, cycle_state: CycleState) -> None:
        self.cycles.append(cycle_state)
        self.current_cycle = cycle_state.cycle
        if cycle_state.plan:
            for task in cycle_state.plan.tasks:
                if task.status == "completed":
                    self.completed_tasks.append(task)
                elif task.status == "failed":
                    self.failed_tasks.append(task)

    def get_all_tasks(self) -> list[TaskModel]:
        tasks = []
        for cycle in self.cycles:
            if cycle.plan:
                tasks.extend(cycle.plan.tasks)
        return tasks


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-sonnet-4-20250514"
    tester_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    state_file: Path = Field(default_factory=lambda: Path.cwd() / ".furrow_state.json")


settings = Settings()
