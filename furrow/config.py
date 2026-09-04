from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    """Supported LLM providers (anthropic, openai, ollama)."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskModel(BaseModel):
    """A single task produced by the planner agent.

    Attributes:
        id: Stable identifier for the task within a plan.
        description: Human-readable description of the work.
        files: Files the task is expected to read or modify.
        dependencies: IDs of tasks that must complete before this one.
        status: Lifecycle status (pending, in_progress, completed, failed).
        result: Free-form result text produced by the worker.
    """

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None


class Plan(BaseModel):
    """Structured plan produced by the planner agent.

    Attributes:
        tasks: Ordered list of tasks to execute in this cycle.
        rationale: Planner's reasoning for the chosen decomposition.
    """

    tasks: list[TaskModel] = Field(default_factory=list)
    rationale: str = ""


class TestResult(BaseModel):
    """Outcome of running the project's test suite after a cycle.

    Attributes:
        passed: True if all tests passed.
        summary: Short human-readable summary.
        failures: Detailed failure messages when tests did not pass.
    """

    passed: bool
    summary: str = ""
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables.

    Environment variables follow the ``FURROW_*`` convention
    (e.g. ``FURROW_PROVIDER``, ``FURROW_MODEL``, ``FURROW_MAX_CYCLES``).
    """

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

    model_config = SettingsConfigDict(
        env_prefix="FURROW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()


def configure_logging(level: str | None = None) -> None:
    """Configure the ``furrow`` logger from settings."""
    log_level = (level or settings.log_level).upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


configure_logging()