from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "Plan",
    "Provider",
    "Settings",
    "TaskModel",
    "TaskStatus",
    "TestResult",
    "get_settings",
    "reset_settings",
]


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


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
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = 5
    # 0 means unlimited: the orchestrator only stops when the goal is satisfied.
    max_cycles: int = Field(
        default=0, description="Maximum orchestration cycles; 0 means unlimited."
    )
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    max_tokens: int = Field(default=8192, description="Max tokens generated per LLM response.")
    request_timeout: float = Field(
        default=120.0, description="Per-request HTTP timeout in seconds."
    )
    max_retries: int = Field(
        default=3, ge=0, description="Max attempts per LLM call; 1 means no retrying."
    )
    test_timeout: float = Field(
        default=300.0, description="Hard timeout in seconds for a test run."
    )


_settings: Settings | None = None


def get_settings(**overrides: Any) -> Settings:
    """Return the process-wide settings, building them on first use.

    Keyword arguments rebuild and re-cache the settings with those values applied, which lets
    tests and the CLI override configuration after import. No validation beyond pydantic's
    happens here, so a missing API key is only reported when a client actually needs it.
    """
    global _settings
    if _settings is None or overrides:
        _settings = Settings(**overrides)
    return _settings


def reset_settings() -> None:
    """Clear the cached Settings so the next get_settings() re-reads the env."""
    global _settings
    _settings = None


def __getattr__(name: str) -> Any:
    """Expose ``settings`` lazily (PEP 562) so importing furrow never reads the environment."""
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), "settings"})
