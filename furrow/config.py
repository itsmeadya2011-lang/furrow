from __future__ import annotations

import json
import os
from datetime import datetime
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
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    cycle: int = 0


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class State(BaseModel):
    goal: str
    tasks: list[TaskModel] = Field(default_factory=list)
    cycles: int = 0
    current_cycle: int = 0
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    def get_pending_tasks(self) -> list[TaskModel]:
        return [t for t in self.tasks if t.status == TaskStatus.PENDING]

    def get_in_progress_tasks(self) -> list[TaskModel]:
        return [t for t in self.tasks if t.status == TaskStatus.IN_PROGRESS]

    def get_completed_tasks(self) -> list[TaskModel]:
        return [t for t in self.tasks if t.status == TaskStatus.COMPLETED]

    def get_failed_tasks(self) -> list[TaskModel]:
        return [t for t in self.tasks if t.status == TaskStatus.FAILED]

    def get_ready_tasks(self) -> list[TaskModel]:
        """Get tasks that are pending and have all dependencies completed."""
        completed_ids = {t.id for t in self.get_completed_tasks()}
        ready = []
        for task in self.get_pending_tasks():
            if all(dep in completed_ids for dep in task.dependencies):
                ready.append(task)
        return ready

    def is_goal_complete(self) -> bool:
        """Check if all tasks are completed and no failures."""
        if any(t.status == TaskStatus.FAILED for t in self.tasks):
            return False
        if not self.tasks:
            return False
        return all(t.status == TaskStatus.COMPLETED for t in self.tasks)

    def to_file(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))

    @classmethod
    def from_file(cls, path: Path) -> State | None:
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls(**data)
        except (OSError, ValueError):
            return None


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
    state_file: Path = Field(default_factory=lambda: Path.cwd() / ".furrow_state.json")

    @field_validator("max_parallel_tasks")
    @classmethod
    def validate_max_parallel(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_parallel_tasks must be at least 1")
        if v > 20:
            raise ValueError("max_parallel_tasks should not exceed 20")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v.upper()

    def validate_config(self) -> list[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []
        
        if self.provider == Provider.ANTHROPIC:
            if not (self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")):
                issues.append("ANTHROPIC_API_KEY not set")
        elif self.provider == Provider.OPENAI:
            if not (self.openai_api_key or os.getenv("OPENAI_API_KEY")):
                issues.append("OPENAI_API_KEY not set")
        elif self.provider == Provider.OLLAMA:
            # Could check if ollama is running
            pass
        
        if self.max_cycles < 0:
            issues.append("max_cycles cannot be negative")
        
        if not self.workspace.exists():
            issues.append(f"Workspace does not exist: {self.workspace}")
        
        return issues


settings = Settings()


def setup_logging() -> None:
    """Configure structured logging with structlog."""
    import logging
    import structlog
    
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
    )
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str):
    """Get a structured logger instance."""
    import structlog
    return structlog.get_logger(name)
