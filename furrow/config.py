from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

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
    # Name collides with pytest's `Test*` collection rule; opt out explicitly.
    __test__ = False

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
    request_timeout: float = 120.0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()


def _resolve_log_level(level: str | None) -> int:
    """Map a level name such as ``"debug"`` to its numeric stdlib level.

    Falls back to ``logging.INFO`` for anything unrecognised instead of raising.
    """
    name = (level or "").strip().upper()
    if not name:
        return logging.INFO
    candidate = getattr(logging, name, None)
    if isinstance(candidate, int):
        return candidate
    resolved = logging.getLevelName(name)
    if isinstance(resolved, int):
        return resolved
    return logging.INFO


def configure_logging(level: str | None = None) -> None:
    """Configure stdlib logging and structlog.

    Uses ``level`` when given, otherwise ``settings.log_level``. Safe to call
    repeatedly and never raises on an invalid level string.
    """
    import structlog

    numeric_level = _resolve_log_level(level if level is not None else settings.log_level)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # basicConfig is a no-op once the root logger has handlers, so this is idempotent.
    logging.basicConfig(level=numeric_level, format="%(message)s")

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
