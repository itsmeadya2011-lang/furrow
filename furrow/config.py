from __future__ import annotations

import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from furrow.costs import ModelPricing

logger = logging.getLogger(__name__)


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None
    error: str | None = None
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    files_changed: list[str] = Field(default_factory=list)

    @field_validator("status", mode="before")
    @classmethod
    def _normalize_status(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "pending"
        known = {"pending", "running", "completed", "failed", "blocked", "skipped"}
        s = v.strip().lower()
        if s in known:
            return s
        return "pending"


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    __test__ = False
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    skipped: bool = False


class WorkerResult(BaseModel):
    summary: str = ""
    success: bool = True
    files_changed: list[str] = Field(default_factory=list)
    error: str | None = None
    iterations: int = 0


class Usage(BaseModel):
    """Token usage accumulated over one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across the counted calls."""
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> Usage:
        """Accumulate *other* into this usage and return self."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.cost_usd += other.cost_usd
        return self


class RunResult(BaseModel):
    passed: bool = False
    cycles: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_blocked: int = 0
    tests_run: int = 0
    summary: str = ""
    stopped_reason: str = ""
    skipped: bool = False
    usage: Usage = Field(default_factory=Usage)
    files_changed: list[str] = Field(default_factory=list)


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
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    max_worker_iterations: int = Field(default=12, ge=1)
    max_retries: int = Field(default=3, ge=0)
    max_task_retries: int = Field(default=1, ge=0)
    request_timeout: float = Field(default=120.0, gt=0)
    command_timeout: float = Field(default=300.0, gt=0)
    test_timeout: float = Field(default=600.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    state_file: Path = Path(".furrow/state.json")
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_settings(self) -> "Settings":
        """Validate settings for consistency and warn about potential issues."""
        # Warn if provider is Anthropic but no API key is set
        if self.provider == Provider.ANTHROPIC and not self.anthropic_api_key:
            import os
            if not os.getenv("ANTHROPIC_API_KEY"):
                warnings.warn(
                    "ANTHROPIC_API_KEY not set. Anthropic provider will fail at runtime. "
                    "Set FURROW_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY environment variable.",
                    UserWarning,
                    stacklevel=2
                )
        
        # Warn if provider is OpenAI but no API key is set
        if self.provider == Provider.OPENAI and not self.openai_api_key:
            import os
            if not os.getenv("OPENAI_API_KEY"):
                warnings.warn(
                    "OPENAI_API_KEY not set. OpenAI provider will fail at runtime. "
                    "Set FURROW_OPENAI_API_KEY or OPENAI_API_KEY environment variable.",
                    UserWarning,
                    stacklevel=2
                )
        
        # Validate log level
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_log_levels:
            warnings.warn(
                f"Invalid log_level '{self.log_level}'. Valid values: {', '.join(valid_log_levels)}. "
                f"Defaulting to INFO.",
                UserWarning,
                stacklevel=2
            )
            self.log_level = "INFO"
        
        # Warn if max_cycles is 0 (unlimited) but max_task_retries is also 0
        if self.max_cycles == 0 and self.max_task_retries == 0:
            warnings.warn(
                "max_cycles=0 (unlimited) with max_task_retries=0 may cause infinite loops "
                "on failing tasks. Consider setting max_task_retries > 0.",
                UserWarning,
                stacklevel=2
            )
        
        # Validate timeouts make sense
        if self.command_timeout < self.request_timeout:
            warnings.warn(
                f"command_timeout ({self.command_timeout}s) is less than request_timeout "
                f"({self.request_timeout}s). Commands may be killed before LLM requests complete.",
                UserWarning,
                stacklevel=2
            )
        
        # Ensure state_file is relative
        if self.state_file.is_absolute():
            warnings.warn(
                f"state_file should be a relative path, got absolute path: {self.state_file}. "
                f"Using relative path '.furrow/state.json'.",
                UserWarning,
                stacklevel=2
            )
            self.state_file = Path(".furrow/state.json")
        
        return self

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)


settings = Settings()
