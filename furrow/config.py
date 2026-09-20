from __future__ import annotations

import unicodedata
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


TaskStatus = Literal["pending", "running", "completed", "failed", "blocked"]


def _validate_safe_relative_path(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("file paths must be non-empty strings")
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("file paths must not contain control characters")
    if "~" in value:
        raise ValueError("file paths must not contain '~'")
    if "\\" in value:
        raise ValueError("file paths must not use Windows separators")
    if len(value) > 1 and value[1] == ":" and value[0].isalpha():
        raise ValueError("file paths must not use Windows drive paths")

    path = PurePosixPath(value)
    if path.is_absolute() or value == "." or any(part == ".." for part in path.parts):
        raise ValueError("file paths must be relative and stay within the workspace")


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = "pending"
    result: Optional[str] = None

    @field_validator("id", "description", mode="before")
    @classmethod
    def strip_required_text(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("task id and description must be non-empty strings")
        return value.strip()

    @field_validator("files", mode="before")
    @classmethod
    def validate_files(cls, value: object) -> list[str]:
        if not isinstance(value, (list, tuple)):
            raise ValueError("files must be a list of paths")
        paths = list(value)
        for path in paths:
            _validate_safe_relative_path(path)
        return paths

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        if any(not isinstance(dependency, str) or not dependency.strip() for dependency in value):
            raise ValueError("task dependencies must contain non-empty ids")
        dependencies = [dependency.strip() for dependency in value]
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("task dependencies must be unique")
        return dependencies


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str

    @field_validator("rationale", mode="before")
    @classmethod
    def strip_rationale(cls, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("plan rationale must be a non-empty string")
        return value.strip()

    @model_validator(mode="after")
    def validate_dependency_graph(self) -> Plan:
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("plan task ids must be unique")

        known_ids = set(task_ids)
        for task in self.tasks:
            missing = [dependency for dependency in task.dependencies if dependency not in known_ids]
            if missing:
                raise ValueError(
                    f"task {task.id!r} has unknown dependencies: {', '.join(missing)}"
                )

        # Kahn's algorithm detects cycles without recursing through a large task graph.
        dependencies = {task.id: list(task.dependencies) for task in self.tasks}
        remaining = {task_id: len(dependencies[task_id]) for task_id in task_ids}
        dependents = {task_id: [] for task_id in task_ids}
        for task_id, task_dependencies in dependencies.items():
            for dependency in task_dependencies:
                dependents[dependency].append(task_id)

        ready = [task_id for task_id, count in remaining.items() if count == 0]
        processed = 0
        while ready:
            task_id = ready.pop()
            processed += 1
            for dependent in dependents[task_id]:
                remaining[dependent] -= 1
                if remaining[dependent] == 0:
                    ready.append(dependent)

        if processed != len(self.tasks):
            cyclic = [task_id for task_id, count in remaining.items() if count > 0]
            raise ValueError(f"plan dependency graph contains a cycle: {', '.join(cyclic)}")
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
    max_parallel_tasks: int = Field(default=5, ge=1)
    # Keep autonomous execution bounded by default so a bad plan cannot loop forever.
    max_cycles: int = Field(default=10, ge=1)
    test_timeout: float = Field(default=120.0, gt=0)
    max_test_output_chars: int = Field(default=10_000, ge=1000)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()
