from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

import structlog
from pydantic import BaseModel, Field
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
    RETRYING = "retrying"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None


class TaskHistoryEntry(BaseModel):
    task_id: str
    description: str
    status: TaskStatus
    result: Optional[str] = None
    cycle: int
    timestamp: datetime


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class State(BaseModel):
    goal: str
    cycles: int
    tasks: list[TaskHistoryEntry]
    last_updated: datetime


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
    state_file: Path = Field(default_factory=lambda: Path.cwd() / ".furrow" / "state.json")
    max_retries: int = 3
    task_timeout: int = 120


settings = Settings()


def ensure_state_dir() -> None:
    settings.state_file.parent.mkdir(parents=True, exist_ok=True)


def save_state(state: State) -> None:
    ensure_state_dir()
    data = state.model_dump(mode="json", exclude_none=True)
    with open(settings.state_file, "w") as f:
        json.dump(data, f, indent=2)


def load_state() -> State | None:
    if not settings.state_file.exists():
        return None
    with open(settings.state_file) as f:
        data = json.load(f)
    return State.model_validate(data)


def setup_logging() -> structlog.types.FilterableLogger:
    import logging
    import sys

    numeric_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    log_file = settings.state_file.parent / "furrow.log"
    ensure_state_dir()
    file_handler = logging.FileHandler(log_file)

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        handlers=[console_handler, file_handler],
    )

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=True,
    )

    return structlog.get_logger()
