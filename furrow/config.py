from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import ClassVar

import structlog
from pydantic import BaseModel, Field
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


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    # Opt out of pytest collection: the name matches its default "Test*" pattern,
    # but this is a pydantic model, not a test class.
    __test__: ClassVar[bool] = False

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
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()

LOG_LEVELS: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "FATAL": logging.FATAL,
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


def configure_logging(level: str | None = None) -> None:
    """Configure structlog, defaulting to ``settings.log_level``.

    Args:
        level: Case-insensitive log level name. Unknown names fall back to INFO.
    """
    resolved = LOG_LEVELS.get((level or settings.log_level).upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(resolved),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
