from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
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
    retry_count: int = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class State(BaseModel):
    """Persistent state for the Furrow orchestrator."""
    goal: str
    cycles: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    completed_tasks: list[TaskModel] = Field(default_factory=list)
    failed_tasks: list[TaskModel] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Path) -> "State | None":
        if not path.exists():
            return None
        return cls.model_validate_json(path.read_text())

    def compact(self, max_completed: int = 20) -> None:
        """Compact state by removing old completed task details."""
        if len(self.completed_tasks) > max_completed:
            # Keep only the most recent completed tasks
            self.completed_tasks = self.completed_tasks[-max_completed:]
        # Clear results from old completed tasks to save space
        for task in self.completed_tasks[:-5]:
            task.result = None


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
    state_file: str = ".furrow_state.json"

    @field_validator("max_parallel_tasks")
    @classmethod
    def validate_max_parallel(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_parallel_tasks must be at least 1")
        if v > 20:
            raise ValueError("max_parallel_tasks cannot exceed 20")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


settings = Settings()
