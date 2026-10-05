from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskModel(BaseModel):
    """One unit of work in a :class:`Plan`."""

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: Literal["pending", "completed", "failed"] = "pending"
    attempts: int = 0
    error: Optional[str] = None
    result: Optional[str] = None


class Plan(BaseModel):
    """A planner-produced decomposition of a goal into tasks."""

    tasks: list[TaskModel]
    rationale: str = ""


class TestResult(BaseModel):
    """Outcome of running the test suite for a task."""

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    command: list[str] = Field(default_factory=list)
    exit_code: Optional[int] = None


class Settings(BaseSettings):
    """Configuration for a furrow run.

    Provider-aware model defaults: when ``provider`` is ``openai`` or
    ``ollama`` and a model field still holds its Anthropic default, the
    provider-appropriate default is substituted.  An explicit value always
    wins, so ``FURROW_MODEL=gpt-4o`` is never overwritten.

    Anthropic defaults: ``claude-sonnet-4-20250514`` /
    ``claude-3-5-haiku-20241022`` / ``claude-3-5-sonnet-20241022``.
    OpenAI defaults: ``gpt-4o`` / ``gpt-4o-mini`` / ``gpt-4o``.
    Ollama defaults: ``qwen2.5-coder`` for all four roles.
    """

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
    max_task_attempts: int = 3
    stagnation_cycles: int = 3
    task_timeout_seconds: float = 600.0
    llm_timeout_seconds: float = 120.0
    llm_max_retries: int = 3
    llm_max_tokens: int = 4096
    test_timeout_seconds: float = 180.0
    workspace: Path = Field(default_factory=Path.cwd)
    state_dir: Optional[Path] = None
    log_level: str = "INFO"

    # --- validation ---------------------------------------------------------

    @field_validator("max_parallel_tasks")
    @classmethod
    def _positive_parallel(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_parallel_tasks must be >= 1")
        return value

    @field_validator("max_task_attempts")
    @classmethod
    def _positive_attempts(cls, value: int) -> int:
        if value < 1:
            raise ValueError("max_task_attempts must be >= 1")
        return value

    @field_validator("stagnation_cycles")
    @classmethod
    def _positive_stagnation(cls, value: int) -> int:
        if value < 1:
            raise ValueError("stagnation_cycles must be >= 1")
        return value

    @field_validator("llm_max_retries")
    @classmethod
    def _non_negative_retries(cls, value: int) -> int:
        if value < 0:
            raise ValueError("llm_max_retries must be >= 0")
        return value

    @field_validator(
        "task_timeout_seconds",
        "llm_timeout_seconds",
        "test_timeout_seconds",
    )
    @classmethod
    def _positive_timeout(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout values must be > 0")
        return value

    # --- provider-aware defaults -------------------------------------------

    @model_validator(mode="after")
    def _provider_model_defaults(self) -> "Settings":
        """Substitute provider defaults when a model field is still the Anthropic one."""
        if self.provider == Provider.OPENAI:
            defaults = {
                "model": "gpt-4o",
                "planner_model": "gpt-4o-mini",
                "worker_model": "gpt-4o",
                "tester_model": "gpt-4o",
            }
        elif self.provider == Provider.OLLAMA:
            defaults = {
                "model": "qwen2.5-coder",
                "planner_model": "qwen2.5-coder",
                "worker_model": "qwen2.5-coder",
                "tester_model": "qwen2.5-coder",
            }
        else:
            return self
        for field_name, default_value in defaults.items():
            current = getattr(self, field_name)
            if current == self.model_fields[field_name].default:
                setattr(self, field_name, default_value)
        return self

    # --- helpers ------------------------------------------------------------

    @property
    def resolved_state_dir(self) -> Path:
        """Return ``state_dir`` when set, else ``<workspace>/.furrow``."""
        if self.state_dir is not None:
            return self.state_dir
        return self.workspace / ".furrow"

    def with_overrides(self, **kwargs: Any) -> "Settings":
        """Return a deep copy with the given fields replaced (singleton is untouched)."""
        return self.model_copy(update=kwargs, deep=True)


def _safe_settings() -> Settings:
    """Build a default-valued :class:`Settings` without touching the environment."""
    try:
        return Settings(_env_file=None)
    except Exception:
        return Settings.model_construct()


settings_load_error: Optional[str] = None
try:
    settings = Settings()
except Exception as exc:  # import must never crash; fall back to safe defaults
    settings_load_error = f"{type(exc).__name__}: {exc}"
    settings = _safe_settings()