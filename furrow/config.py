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


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

    # Lowercase aliases keep the enum convenient for callers that use status names directly.
    pending = PENDING
    running = RUNNING
    completed = COMPLETED
    failed = FAILED


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task id must not be empty")
        return value

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        if any(not dependency.strip() for dependency in value):
            raise ValueError("dependency ids must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("task dependencies must not contain duplicates")
        return value


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str

    @model_validator(mode="after")
    def validate_ids(self) -> "Plan":
        task_ids = [task.id for task in self.tasks]
        seen_ids: set[str] = set()
        duplicates: list[str] = []
        for task_id in task_ids:
            if task_id in seen_ids:
                duplicates.append(task_id)
            seen_ids.add(task_id)
        if duplicates:
            raise ValueError(f"duplicate task id: {duplicates[0]!r}")

        for task in self.tasks:
            if task.id in task.dependencies:
                raise ValueError(f"task {task.id!r} depends on itself")
        return self

    def validate_graph(self, known_ids: set[str] | None = None) -> None:
        task_ids = [task.id for task in self.tasks]
        seen_ids: set[str] = set()
        duplicates: list[str] = []
        for task_id in task_ids:
            if task_id in seen_ids:
                duplicates.append(task_id)
            seen_ids.add(task_id)
        if duplicates:
            raise ValueError(f"duplicate task id: {duplicates[0]!r}")

        all_ids = set(task_ids)
        if known_ids is not None:
            all_ids.update(known_ids)
        for task in self.tasks:
            if task.id in task.dependencies:
                raise ValueError(f"task {task.id!r} depends on itself")
            for dependency in task.dependencies:
                if dependency not in all_ids:
                    raise ValueError(
                        f"task {task.id!r} has unknown dependency {dependency!r}"
                    )


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
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()
