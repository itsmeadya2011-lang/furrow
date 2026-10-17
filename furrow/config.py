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


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class CycleState(BaseModel):
    """Persistent state for a single orchestrator cycle."""

    cycle: int
    goal: str
    plan: Optional[Plan] = None
    task_results: dict[str, str] = Field(default_factory=dict)
    test_result: Optional[TestResult] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class OrchestratorState(BaseModel):
    """Persistent state for the orchestrator across cycles/sessions."""

    goal: str
    cycles: int = 0
    completed: bool = False
    cycle_states: list[CycleState] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "OrchestratorState":
        p = Path(path)
        if not p.exists():
            return cls(goal="")
        data = json.loads(p.read_text())
        return cls.model_validate(data)


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
    ollama_model: str = "llama3.1"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    state_file: Path = Field(default_factory=lambda: Path.cwd() / ".furrow_state.json")
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "console"


settings = Settings()
