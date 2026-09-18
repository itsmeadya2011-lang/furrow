from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
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

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Task id must be non-empty")
        return value

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        if any(not dependency.strip() for dependency in value):
            raise ValueError("Task dependencies must contain non-empty ids")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        allowed = {"pending", "running", "completed", "failed", "blocked"}
        if value not in allowed:
            raise ValueError(f"Task status must be one of {sorted(allowed)}")
        return value

    @model_validator(mode="after")
    def validate_self_dependency(self) -> TaskModel:
        if self.id in self.dependencies:
            raise ValueError(f"Task {self.id!r} cannot depend on itself")
        return self


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str

    @model_validator(mode="after")
    def validate_tasks(self) -> Plan:
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Plan task ids must be unique")
        known_ids = set(ids)
        for task in self.tasks:
            missing = [dependency for dependency in task.dependencies if dependency not in known_ids]
            if missing:
                raise ValueError(
                    f"Task {task.id!r} has unknown dependencies: {', '.join(missing)}"
                )
        return self


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
    test_command: Optional[str] = None
    test_timeout: int = 120
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    @field_validator("max_parallel_tasks")
    @classmethod
    def validate_max_parallel_tasks(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("max_parallel_tasks must be greater than 0")
        return value

    @field_validator("max_cycles")
    @classmethod
    def validate_max_cycles(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_cycles must be greater than or equal to 0")
        return value

    @field_validator("test_timeout")
    @classmethod
    def validate_test_timeout(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("test_timeout must be greater than 0")
        return value

    @field_validator("test_command")
    @classmethod
    def validate_test_command(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("test_command must be non-empty when provided")
        return value

    @field_validator("workspace", mode="before")
    @classmethod
    def validate_workspace(cls, value: str | Path) -> Path:
        if not str(value).strip():
            raise ValueError("workspace must be non-empty")
        try:
            workspace = Path(value).expanduser().resolve(strict=False)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("workspace must resolve to an existing path") from exc
        if not workspace.is_dir():
            raise ValueError("workspace must be an existing directory")
        return workspace


settings = Settings()
