from __future__ import annotations

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


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env", extra="ignore")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-3-5-sonnet-20241022"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = 5
    max_cycles: int = 10
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    @field_validator("max_parallel_tasks")
    @classmethod
    def validate_max_parallel(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_parallel_tasks must be at least 1")
        if v > 20:
            raise ValueError("max_parallel_tasks must not exceed 20")
        return v

    @field_validator("max_cycles")
    @classmethod
    def validate_max_cycles(cls, v: int) -> int:
        if v < 0:
            raise ValueError("max_cycles must be non-negative")
        if v > 100:
            raise ValueError("max_cycles must not exceed 100")
        return v


settings = Settings()