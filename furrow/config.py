"""Domain models and application settings for Furrow."""
from __future__ import annotations

import functools
import logging
import os
import sys
from enum import Enum
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: The logging levels ``log_level`` accepts. Spelled once so the field
#: validator, :func:`configure_logging`, the CLI option and ``.env.example``
#: cannot drift apart again.
LOG_LEVELS: Final[tuple[str, ...]] = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


class Provider(str, Enum):
    """LLM providers supported as the backing model service."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Lifecycle states a task transitions through while running."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    """A single unit of work produced by the planner."""

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


class Plan(BaseModel):
    """An ordered collection of tasks plus the reasoning behind them."""

    tasks: list[TaskModel] = Field(default_factory=list)
    rationale: str = ""


class TestResult(BaseModel):
    """Outcome of executing the test suite for a task or the workspace."""

    __test__ = False

    passed: bool
    summary: str = ""
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Runtime configuration populated from ``FURROW_*`` env vars and ``.env``.

    ``validate_assignment`` is on so the :func:`_validate_log_level` invariant also
    holds on the mutation path: the CLI assigns ``settings.log_level`` directly
    (:func:`furrow.cli.main.start`), and without this an unknown level would be
    accepted there while being rejected at construction.
    """

    model_config = SettingsConfigDict(
        env_prefix="FURROW_",
        env_file=".env",
        validate_assignment=True,
    )

    provider: Provider = Field(
        default=Provider.ANTHROPIC,
        description="LLM provider: anthropic, openai, or ollama.",
    )
    model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Default model used for general completion.",
    )
    planner_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Model used by the planning stage.",
    )
    worker_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Model used to execute task work.",
    )
    tester_model: str = Field(
        default="claude-3-5-sonnet-20241022",
        description="Model used by the testing stage.",
    )
    override_model: str | None = Field(
        default=None,
        description="If set, wins over every role-specific model.",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key; falls back to ANTHROPIC_API_KEY.",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key; falls back to OPENAI_API_KEY.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the local Ollama service.",
    )
    max_parallel_tasks: int = Field(
        default=5,
        description="Maximum number of tasks executed in parallel.",
    )
    max_cycles: int = Field(
        default=0,
        description="Maximum planning/execution cycles; 0 means unlimited.",
    )
    max_tokens: int = Field(
        default=4096,
        description="Default maximum tokens for LLM requests.",
    )
    request_timeout: float = Field(
        default=120.0,
        description="HTTP timeout in seconds for LLM requests.",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries on transient LLM failures.",
    )
    test_timeout: float = Field(
        default=120.0,
        description="Timeout in seconds for running test commands.",
    )
    test_command: str | None = Field(
        default=None,
        description="Explicit test command override; auto-detected when unset.",
    )
    workspace: Path = Field(
        default_factory=Path.cwd,
        description="Directory treated as the project workspace.",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging verbosity: DEBUG, INFO, WARNING, ERROR, or CRITICAL.",
    )

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        """Normalise ``log_level`` to a canonical upper-case stdlib level.

        Args:
            value: The raw value, from the environment, a ``.env`` file, or a
                constructor argument.

        Returns:
            The upper-cased level name.

        Raises:
            ValueError: If the value is not one of :data:`LOG_LEVELS`.
        """
        return normalize_log_level(value)

    def api_key_for(self, provider: Provider | None = None) -> str | None:
        """Resolve the API key for a provider, falling back to process env.

        Uses ``provider`` when given, otherwise the configured
        :attr:`provider`. Returns ``None`` for Ollama, which needs no key.
        """
        resolved = provider if provider is not None else self.provider
        if resolved == Provider.ANTHROPIC:
            return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if resolved == Provider.OPENAI:
            return self.openai_api_key or os.getenv("OPENAI_API_KEY")
        return None


class ConfigurationError(ValueError):
    """Raised when :class:`Settings` cannot be built from the configuration.

    :func:`get_settings` runs at import time, so an unusable ``FURROW_LOG_LEVEL``
    in a ``.env`` file otherwise surfaces as a raw
    :class:`pydantic.ValidationError` traceback out of ``import furrow`` with no
    hint that the operator's configuration is at fault. This is a ``ValueError``
    so existing ``except ValueError`` handling of pydantic validation keeps
    working.
    """


def _describe_invalid_settings(exc: ValidationError) -> str:
    """Summarise which settings failed validation, as ``field=value`` pairs.

    Args:
        exc: The validation error raised while building :class:`Settings`.

    Returns:
        A single-line, human-readable list of the offending fields and values.
    """
    return ", ".join(
        f"{'.'.join(str(part) for part in error['loc'])}={error.get('input')!r}"
        for error in exc.errors()
    )


def _configuration_error_message(exc: ValidationError) -> str:
    """Render the operator-facing explanation for a bad configuration.

    This is the *first* thing a user sees, because :func:`get_settings` runs at
    import time: a bare ``pydantic.ValidationError`` traceback out of
    ``import furrow`` names an internal type and not the file to edit. The
    message therefore names both sources, the offending value, and where the full
    pydantic report lives.

    Args:
        exc: The validation error raised while building :class:`Settings`.

    Returns:
        A short, multi-line message suitable for a terminal.
    """
    env_file = Path(str(Settings.model_config.get("env_file", ".env"))).resolve()
    return (
        f"Furrow could not load its configuration: {_describe_invalid_settings(exc)}.\n"
        f"Settings come from FURROW_* environment variables and the .env file in "
        f"the working directory ({env_file}); that value is not valid.\n"
        f"Fix or remove the offending value and re-run; the full pydantic report "
        f"is chained below this message."
    )


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Build and cache the global :class:`Settings` singleton.

    Construction is centralised here and cached so every consumer — including
    the module-level ``settings`` — shares a single instance.

    Raises:
        ConfigurationError: If the environment or ``.env`` file supplies a value
            that is not valid. The originating :class:`pydantic.ValidationError`
            is chained as ``__cause__`` so the per-field detail is not lost.
    """
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigurationError(_configuration_error_message(exc)) from exc


def normalize_log_level(level: str) -> str:
    """Return `level` as a canonical upper-case stdlib level name.

    Args:
        level: A raw level name such as ``"debug"``.

    Returns:
        The upper-cased level name.

    Raises:
        ValueError: If the value is not one of :data:`LOG_LEVELS`.
    """
    normalized = str(level).strip().upper()
    if normalized not in LOG_LEVELS:
        raise ValueError(
            f"log_level must be one of {', '.join(LOG_LEVELS)}; got {level!r}"
        )
    return normalized


def configure_logging(level: str | None = None) -> str:
    """Apply `level`, or ``FURROW_LOG_LEVEL``, to the root logger.

    This is the single place logging is wired up, so the ``log_level`` setting is
    honoured instead of merely being documented. It is deliberately *not* called
    at import time: importing :mod:`furrow.config` must stay free of side
    effects, so only the CLI entry points call it.

    Args:
        level: Level name to apply, or ``None`` to use the configured
            :attr:`Settings.log_level`.

    Returns:
        The level that was applied.

    Raises:
        ValueError: If the resolved level is not one of :data:`LOG_LEVELS`.

    Notes:
        ``force=True`` is never passed to :func:`logging.basicConfig`: it would
        discard the handlers of a host application that imported this module.
        A handler is therefore installed only when the root logger has none, so
        repeated calls are harmless, and only the root *level* is re-applied.
    """
    resolved = normalize_log_level(
        level if level is not None else get_settings().log_level
    )
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(
            level=resolved,
            stream=sys.stderr,
            format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        )
    root.setLevel(resolved)
    return resolved


settings = get_settings()
