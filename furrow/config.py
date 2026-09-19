from __future__ import annotations

from enum import Enum
from pathlib import Path

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
    result: str | None = None

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Task id must be non-empty")
        return value

    @field_validator("files")
    @classmethod
    def validate_files(cls, value: list[str]) -> list[str]:
        if any(not item.strip() for item in value):
            raise ValueError("Task files must contain non-empty paths")
        return value

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: list[str]) -> list[str]:
        if any(not dependency.strip() for dependency in value):
            raise ValueError("Task dependencies must contain non-empty ids")
        if len(value) != len(set(value)):
            raise ValueError("Task dependencies must be unique")
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

    @field_validator("rationale")
    @classmethod
    def validate_rationale(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Plan rationale must be non-empty")
        return value

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

        graph = {task.id: task.dependencies for task in self.tasks}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise ValueError(f"Plan dependencies contain a cycle at task {task_id!r}")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in graph[task_id]:
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in ids:
            visit(task_id)
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
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    test_command: str | None = None
    test_timeout: float = 120.0
    workspace: Path = Field(default_factory=lambda: Path.cwd().resolve())
    log_level: str = "INFO"
    llm_timeout: float = 120.0
    max_attempts: int = Field(default=2, ge=1)
    retry_delay: float = 1.0

    @field_validator("model", "planner_model", "worker_model", "tester_model")
    @classmethod
    def validate_model_names(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Model names must be non-empty")
        return value

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("ollama_base_url must be non-empty")
        return value.rstrip("/")

    @field_validator("test_command")
    @classmethod
    def validate_test_command(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("test_command must be non-empty when provided")
        return value

    @field_validator("test_timeout", "llm_timeout", "retry_delay")
    @classmethod
    def validate_positive_floats(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Timeout and retry settings must be greater than 0")
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
