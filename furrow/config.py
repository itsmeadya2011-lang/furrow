from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path

import structlog
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LOG_LEVELS: tuple[str, ...] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

DEFAULT_MAX_CONCURRENT_RUNS = 2
"""Default number of furrow runs (web or CLI ``run()``) served at the same time."""


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Lifecycle state of a single planned task.

    Subclasses :class:`str` on purpose: existing callers compare plain strings
    (``task.status == "completed"``), assign them (``task.status = "failed"``)
    and serialise them, and all of that keeps working.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    __str__ = str.__str__
    __repr__ = str.__repr__

    @classmethod
    def _missing_(cls, value: object) -> TaskStatus | None:
        """Accept loose spellings such as ``"Completed"`` or ``"in-progress"``."""
        if not isinstance(value, str):
            return None
        candidate = value.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == candidate:
                return member
        return None

    @property
    def is_terminal(self) -> bool:
        """True when no further transition is expected."""
        return self in _TERMINAL_STATUSES


_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.SKIPPED}
)


class TaskModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    id: str = Field(min_length=1, description="Stable identifier of the task within a plan.")
    description: str = Field(min_length=1, description="What the worker agent must implement.")
    files: list[str] = Field(
        default_factory=list, description="Files the task is expected to touch."
    )
    dependencies: list[str] = Field(
        default_factory=list, description="Ids of tasks that must complete before this one runs."
    )
    status: TaskStatus = Field(
        default=TaskStatus.PENDING, description="Current lifecycle state of the task."
    )
    result: str | None = Field(default=None, description="Outcome produced by the worker, if any.")

    @field_validator("id", "description", mode="before")
    @classmethod
    def _strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class Plan(BaseModel):
    tasks: list[TaskModel] = Field(description="Ordered, dependency-aware units of work.")
    rationale: str = Field(description="Model's explanation of how the plan reaches the goal.")


class TestResult(BaseModel):
    passed: bool = Field(description="True when the test suite passed.")
    summary: str = Field(description="Short human readable summary of the run.")
    failures: list[str] = Field(default_factory=list, description="Individual failure messages.")


def _normalise_log_level(value: object) -> str:
    """Upper-case and validate a log level name."""
    candidate = value.strip().upper() if isinstance(value, str) else str(value).strip().upper()
    if candidate not in LOG_LEVELS:
        raise ValueError(
            f"log_level must be one of {', '.join(LOG_LEVELS)} (case-insensitive), got {value!r}"
        )
    return candidate


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Field(
        default=Provider.ANTHROPIC,
        description="LLM backend used for completions (anthropic, openai or ollama).",
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        min_length=1,
        description="Default model id used when no agent specific override is given.",
    )
    planner_model: str = Field(
        default="claude-3-5-haiku-20241022",
        min_length=1,
        description="Model used by the planner agent to turn a goal into a Plan.",
    )
    worker_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        min_length=1,
        description="Model used by worker agents to implement individual tasks.",
    )
    tester_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        min_length=1,
        description="Model used by the tester agent to interpret a test run.",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key; falls back to the ANTHROPIC_API_KEY env var.",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key; falls back to the OPENAI_API_KEY env var.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        min_length=1,
        description="Base URL of the Ollama server used when provider is 'ollama'.",
    )
    max_parallel_tasks: int = Field(
        default=5,
        ge=1,
        description="Maximum number of tasks executed concurrently; must be at least 1.",
    )
    max_concurrent_runs: int = Field(
        default=DEFAULT_MAX_CONCURRENT_RUNS,
        ge=1,
        description=(
            "Maximum number of whole furrow runs executing concurrently; must be at least 1. "
            "The web layer still honours its own FURROW_WEB_MAX_CONCURRENCY override."
        ),
    )
    max_cycles: int = Field(
        default=0,
        ge=0,
        description="Maximum number of development cycles; 0 means unlimited.",
    )
    workspace: Path = Field(
        default_factory=Path.cwd, description="Root directory furrow reads and writes files in."
    )
    log_level: str = Field(
        default="INFO",
        description="Minimum level that gets logged: DEBUG, INFO, WARNING, ERROR or CRITICAL.",
    )
    test_command: str | None = Field(
        default=None,
        description="Explicit shell-style test command; when unset the tester auto-detects one.",
    )
    test_timeout: int = Field(
        default=120,
        ge=1,
        description="Seconds a single test run may take before it is killed.",
    )
    retry_attempts: int = Field(
        default=3,
        ge=1,
        description="Total LLM attempts (1 = no retry) for transient provider failures.",
    )
    retry_min_wait: float = Field(
        default=1.0,
        ge=0.0,
        description="Base seconds for the exponential backoff between LLM retries.",
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=1,
        description="Maximum number of tokens requested per LLM completion.",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Sampling temperature sent with every LLM completion.",
    )

    @field_validator(
        "model",
        "planner_model",
        "worker_model",
        "tester_model",
        "ollama_base_url",
        mode="before",
    )
    @classmethod
    def _strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("test_command", "anthropic_api_key", "openai_api_key", mode="before")
    @classmethod
    def _strip_optional_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, value: object) -> str:
        return _normalise_log_level(value)


settings = Settings()


def configure_logging(level: str | None = None) -> None:
    """Configure :mod:`structlog` (console renderer, timestamp, level filtering).

    Uses ``settings.log_level`` unless *level* is given. Safe to call repeatedly:
    the most recent call wins and no handler or formatter is ever duplicated.
    """
    resolved = _normalise_log_level(settings.log_level if level is None else level)
    numeric = getattr(logging, resolved, logging.INFO)

    logging.getLogger().setLevel(numeric)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=False,
    )
    structlog.get_logger(__name__).debug("logging_configured", level=resolved)
