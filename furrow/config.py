from __future__ import annotations

from enum import Enum
from pathlib import Path, PureWindowsPath
from typing import Literal

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    StrictStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"

    pending = PENDING
    running = RUNNING
    in_progress = IN_PROGRESS
    completed = COMPLETED
    failed = FAILED
    blocked = BLOCKED


class FileOperation(BaseModel):
    action: Literal["write", "delete"] = Field(
        validation_alias=AliasChoices("action", "operation", "op")
    )
    path: StrictStr
    content: StrictStr | None = None

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: StrictStr) -> StrictStr:
        path = value.strip()
        if not path:
            raise ValueError("file operation path must not be empty")
        if path.startswith("~"):
            raise ValueError("file operation path must be relative to the workspace")
        if path.startswith(("/", "\\")):
            raise ValueError("file operation path must be relative to the workspace")

        windows_path = PureWindowsPath(path)
        if windows_path.is_absolute() or windows_path.drive:
            raise ValueError("file operation path must be relative to the workspace")

        normalized = path.replace("\\", "/")
        if normalized == ".":
            raise ValueError("file operation path must identify a file")
        if any(part == ".." for part in normalized.split("/")):
            raise ValueError("file operation path must not traverse parent directories")
        return path

    @model_validator(mode="after")
    def _validate_shape(self) -> FileOperation:
        if self.action == "write" and self.content is None:
            raise ValueError("write operation requires content")
        if self.action == "delete" and self.content is not None:
            raise ValueError("delete operation must not include content")
        return self


class WriteOperation(FileOperation):
    action: Literal["write"] = "write"
    content: StrictStr


class DeleteOperation(FileOperation):
    action: Literal["delete"] = "delete"
    content: None = None


FileWrite = WriteOperation
FileDelete = DeleteOperation
WriteFileOperation = WriteOperation
DeleteFileOperation = DeleteOperation


class WorkerResult(BaseModel):
    task_id: StrictStr | None = None
    summary: StrictStr = ""
    operations: list[FileOperation] = Field(
        default_factory=list,
        validation_alias=AliasChoices("operations", "files", "changes"),
    )
    success: bool = True
    error: StrictStr | None = None
    issues: list[StrictStr] = Field(default_factory=list)
    files_changed: list[StrictStr] = Field(default_factory=list)
    complete: bool | None = Field(default=None, exclude=True)

    @field_validator("task_id")
    @classmethod
    def _validate_task_id(cls, value: StrictStr | None) -> StrictStr | None:
        if value is not None and not value.strip():
            raise ValueError("worker result task_id must not be empty")
        return value

    @model_validator(mode="after")
    def _normalize_completion(self) -> WorkerResult:
        if self.complete is not None:
            self.success = self.complete
        return self

    @property
    def files(self) -> list[FileOperation]:
        return self.operations

    @property
    def result(self) -> str:
        return self.summary


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task id must not be empty")
        return value.strip()

    @field_validator("description")
    @classmethod
    def _validate_description(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task description must not be empty")
        return value

    @field_validator("files")
    @classmethod
    def _validate_files(cls, value: list[str]) -> list[str]:
        if any(not path.strip() for path in value):
            raise ValueError("task file paths must not be empty")
        return value

    @field_validator("dependencies")
    @classmethod
    def _validate_dependencies(cls, value: list[str]) -> list[str]:
        if any(not dependency.strip() for dependency in value):
            raise ValueError("task dependency ids must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("task dependencies must not contain duplicates")
        return value


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str

    @model_validator(mode="after")
    def _validate_task_ids(self) -> Plan:
        task_ids = [task.id for task in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("task ids must be unique within a plan")
        for task in self.tasks:
            if task.id in task.dependencies:
                raise ValueError(f"task {task.id!r} must not depend on itself")
        return self

    def validate_graph(self, known_ids: set[str] | None = None) -> None:
        task_ids = {task.id for task in self.tasks}
        if known_ids is not None:
            task_ids.update(known_ids)
        for task in self.tasks:
            for dependency in task.dependencies:
                if dependency not in task_ids:
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
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    task_timeout: float = Field(default=120.0, gt=0)
    test_timeout: float = Field(default=120.0, gt=0)
    workspace: Path = Field(default_factory=lambda: Path.cwd().resolve())
    log_level: str = "INFO"

    @field_validator("workspace", mode="before")
    @classmethod
    def _normalize_workspace(cls, value: str | Path) -> Path:
        return Path(value).expanduser().resolve()

    @model_validator(mode="after")
    def _validate_provider(self) -> Settings:
        if self.provider == Provider.OLLAMA:
            raise ValueError("Provider.OLLAMA is not supported by this runtime")
        return self


settings = Settings()
