from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = TaskStatus.PENDING.value
    result: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        # Allow any string for backward compatibility but normalize known statuses
        return v


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    """Result of running tests after a development cycle."""

    __test__ = False  # Prevent pytest from collecting this as a test class

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class OrchestratorState(BaseModel):
    """Persistent state for the orchestrator across cycles and sessions."""

    goal: str
    cycle: int = 0
    task_history: list[TaskModel] = Field(default_factory=list)
    test_history: list[TestResult] = Field(default_factory=list)
    completed: bool = False

    def add_task(self, task: TaskModel) -> None:
        """Record a task in history, replacing any existing task with the same id."""
        self.task_history = [t for t in self.task_history if t.id != task.id]
        self.task_history.append(task.model_copy())

    def update_task(self, task: TaskModel) -> None:
        """Update a task in history by id."""
        for i, t in enumerate(self.task_history):
            if t.id == task.id:
                self.task_history[i] = task.model_copy()
                return
        self.task_history.append(task.model_copy())


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
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    state_file: str = ".furrow-state.json"

    @field_validator("max_parallel_tasks")
    @classmethod
    def validate_max_parallel(cls, v: int) -> int:
        if v < 1:
            return 1
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        level = v.upper()
        if level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(f"Invalid log level: {v}")
        return level


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    import structlog

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer()
            if level.upper() == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(logging, level.upper())),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


settings = Settings()


def get_logger(name: str = "furrow"):
    """Get a structlog logger, initializing with default config if needed."""
    try:
        import structlog

        return structlog.get_logger(name)
    except Exception:
        import logging

        return logging.getLogger(name)
