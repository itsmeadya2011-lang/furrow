from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Directories skipped while discovering project files. A copy is handed to every
# ``Settings`` instance so callers can mutate ``ignored_dirs`` freely.
_DEFAULT_IGNORED_DIRS: tuple[str, ...] = (
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "dist",
    "build",
    "target",
    ".next",
    ".tox",
    "htmlcov",
)


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Lifecycle state of a :class:`TaskModel`.

    Subclasses ``str`` so existing comparisons such as
    ``task.status == "completed"`` keep working, while still constraining the
    field to a closed set of values.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None

    @field_validator("status", mode="before")
    @classmethod
    def _coerce_status(cls, value: Any) -> Any:
        """Accept plain strings like ``"failed"``, but reject typos explicitly."""
        if isinstance(value, TaskStatus):
            return value
        if isinstance(value, str):
            try:
                return TaskStatus(value)
            except ValueError as exc:
                allowed = ", ".join(repr(member.value) for member in TaskStatus)
                raise ValueError(
                    f"Invalid task status {value!r}; expected one of: {allowed}"
                ) from exc
        return value


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Runtime configuration loaded from ``FURROW_*`` env vars and ``.env``."""

    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = 5
    # Any value <= 0 means "unlimited": keep looping until the goal is reached.
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    # Explicit shell-style test command override; None means auto-detect.
    test_command: Optional[str] = None
    # Upper bound on tokens requested per LLM completion.
    max_tokens: int = 4096
    ignored_dirs: list[str] = Field(default_factory=lambda: list(_DEFAULT_IGNORED_DIRS))

    @property
    def anthropic_api_key_text(self) -> Optional[str]:
        """Anthropic API key in plaintext, or ``None`` when unset."""
        if self.anthropic_api_key is None:
            return None
        return self.anthropic_api_key.get_secret_value()

    @property
    def openai_api_key_text(self) -> Optional[str]:
        """OpenAI API key in plaintext, or ``None`` when unset."""
        if self.openai_api_key is None:
            return None
        return self.openai_api_key.get_secret_value()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide :class:`Settings`, constructing it on first use."""
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached :class:`Settings` so the next access re-reads the env."""
    get_settings.cache_clear()


def __getattr__(name: str) -> Any:
    """Resolve ``settings`` lazily (PEP 562).

    Keeps ``from furrow.config import settings`` working without building a
    ``Settings`` instance at import time, so env vars and ``.env`` files are
    picked up on first use. Every access returns the same cached instance.
    """
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
