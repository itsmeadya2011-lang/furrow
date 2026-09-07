from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import structlog
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationInfo
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
    result: Optional[str] = None


class WorkerResult(BaseModel):
    files_changed: list[str] = Field(default_factory=list)
    summary: str = ""
    complete: bool = False


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

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, v: str, info: ValidationInfo) -> str:
        provider = info.data.get("provider") if info.data else None
        if provider == Provider.OLLAMA:
            parsed = urlparse(v)
            if parsed.scheme not in ("http", "https"):
                raise ValueError(f"ollama_base_url must be a valid HTTP/HTTPS URL when using Ollama: {v}")
            if not parsed.hostname:
                raise ValueError(f"ollama_base_url must have a valid hostname when using Ollama: {v}")
        return v

    @field_validator("workspace")
    @classmethod
    def validate_workspace(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"workspace directory does not exist: {v}")
        if not v.is_dir():
            raise ValueError(f"workspace is not a directory: {v}")
        return v


def setup_logging(level: str) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level.upper(),
    )


settings = Settings()
