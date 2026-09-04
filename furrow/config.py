from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator
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
    request_timeout: float = 60.0

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        level = logging.getLevelName(self.log_level)
        if not isinstance(level, int) or level <= 0:
            raise ValueError(f"Invalid log_level: {self.log_level}")
        if self.max_parallel_tasks <= 0:
            raise ValueError("max_parallel_tasks must be > 0")
        if not self.workspace.exists():
            try:
                self.workspace.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                raise ValueError(
                    f"Cannot create workspace directory: {self.workspace}"
                ) from exc
        return self


settings = Settings()
