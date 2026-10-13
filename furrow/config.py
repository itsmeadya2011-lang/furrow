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

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"pending", "completed", "failed", "verified"}
        if v not in valid:
            raise ValueError(f"Invalid status: {v}. Must be one of {valid}")
        return v


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str

    @field_validator("tasks")
    @classmethod
    def validate_tasks(cls, v: list[TaskModel]) -> list[TaskModel]:
        if len(v) > 10:
            raise ValueError("Too many tasks in plan (max 10)")
        return v


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
    ollama_model: str = "llama3.1:8b"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    @field_validator("max_parallel_tasks")
    @classmethod
    def validate_max_parallel_tasks(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_parallel_tasks must be at least 1")
        if v > 20:
            raise ValueError("max_parallel_tasks cannot exceed 20")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"Invalid log_level: {v}. Must be one of {valid}")
        return upper


settings = Settings()


def validate_settings() -> list[str]:
    """Validate settings and return list of warnings/errors."""
    issues = []
    
    if settings.provider == Provider.ANTHROPIC:
        if not settings.anthropic_api_key and not os.getenv("ANTHROPIC_API_KEY"):
            issues.append("ANTHROPIC_API_KEY not set (required for Anthropic provider)")
    elif settings.provider == Provider.OPENAI:
        if not settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY not set (required for OpenAI provider)")
    elif settings.provider == Provider.OLLAMA:
        # Ollama doesn't require API key, just check URL format
        if not settings.ollama_base_url.startswith(("http://", "https://")):
            issues.append(f"Invalid ollama_base_url: {settings.ollama_base_url}")
    
    if not settings.workspace.exists():
        issues.append(f"Workspace does not exist: {settings.workspace}")
    
    return issues


import os
