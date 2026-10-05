"""Configuration, enumerations, and pydantic data models for Furrow.

`Settings` is loaded from the environment (prefix ``FURROW_``) and from an
optional ``.env`` file.  A module level `settings` singleton is exposed for
convenience, but every consumer may build its own instance instead.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "VALID_TASK_STATUSES",
    "Plan",
    "Provider",
    "Settings",
    "TaskModel",
    "TaskStatus",
    "TestResult",
    "settings",
]

DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."


class Provider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Lifecycle states a planned task can be in.

    Members subclass `str`, so ``task.status == "completed"`` keeps working for
    code that compares against plain strings.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


VALID_TASK_STATUSES: frozenset[str] = frozenset(status.value for status in TaskStatus)


class TaskModel(BaseModel):
    """A single unit of work produced by the planner."""

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, value: Any) -> str:
        """Normalize and validate a task status.

        Args:
            value: The raw status, either a `str` or a `TaskStatus` member.

        Returns:
            The lower-cased, stripped status string.

        Raises:
            ValueError: If the status is not one of `VALID_TASK_STATUSES`.
        """
        raw = value.value if isinstance(value, Enum) else value
        normalized = str(raw).strip().lower()
        if normalized not in VALID_TASK_STATUSES:
            valid = ", ".join(sorted(VALID_TASK_STATUSES))
            raise ValueError(f"Invalid task status {raw!r}; expected one of: {valid}")
        return normalized

    @property
    def succeeded(self) -> bool:
        """Return `True` when the task finished with the ``completed`` status."""
        return str(self.status).strip().lower() == TaskStatus.COMPLETED.value


class Plan(BaseModel):
    """An ordered set of tasks produced by the planner agent."""

    tasks: list[TaskModel]
    rationale: str

    def task_by_id(self, task_id: str) -> TaskModel | None:
        """Return the task matching `task_id`, or `None` when it is absent."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None

    @property
    def completed_ids(self) -> frozenset[str]:
        """Return the ids of every task that completed successfully."""
        return frozenset(task.id for task in self.tasks if task.succeeded)


class TestResult(BaseModel):
    """Outcome of one test-suite run."""

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Runtime configuration, populated from the environment and `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="FURROW_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

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
    max_tokens: int = 4096
    max_retries: int = Field(default=3, ge=1)
    retry_base_delay: float = Field(default=1.0, gt=0)
    request_timeout: float = 60.0
    test_timeout: float = 120.0
    default_system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_file_bytes: int = 1_000_000
    ignore_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".kilo",
            "__pycache__",
            ".venv",
            "venv",
            "node_modules",
            ".mypy_cache",
            ".pytest_cache",
            ".ruff_cache",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
        ]
    )
    max_prompt_files: int = 200
    file_context_chars: int = 4000


settings = Settings()
