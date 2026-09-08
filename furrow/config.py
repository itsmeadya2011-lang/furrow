from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import structlog
except ImportError:
    structlog = None


def setup_logging(level: str | None = None) -> None:
    if level is None:
        level = settings.log_level
    logging.basicConfig(level=level, format="%(message)s")
    if structlog is not None:
        shared_processors = [
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
        ]
        structlog.configure(
            processors=[
                *shared_processors,
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        formatter = structlog.stdlib.ProcessorFormatter(
            processors=shared_processors,
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.handlers = [handler]


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


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


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
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {sorted(valid_levels)}, got {v!r}")
        return v_upper


settings = Settings()
setup_logging()
