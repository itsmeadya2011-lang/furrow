from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from furrow.costs import ModelPricing


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


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    def __init__(self, message: str, field: str | None = None):
        self.field = field
        super().__init__(message)


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

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        if isinstance(v, str):
            return v.upper()
        return "INFO"

    @field_validator("workspace", mode="before")
    @classmethod
    def _resolve_workspace(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v).resolve()
        return v.resolve()

    @field_validator("ollama_base_url")
    @classmethod
    def _normalize_ollama_url(cls, v: str) -> str:
        if not v:
            return "http://localhost:11434"
        url = v.rstrip("/")
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return url

    @model_validator(mode="after")
    def _validate_provider_keys(self) -> "Settings":
        """Validate that required API keys are present for the selected provider."""
        if self.provider == Provider.ANTHROPIC:
            if not self.anthropic_api_key and not self._get_env("ANTHROPIC_API_KEY"):
                raise ConfigValidationError(
                    "ANTHROPIC_API_KEY is required when using the Anthropic provider. "
                    "Set it via FURROW_ANTHROPIC_API_KEY environment variable or .env file.",
                    "anthropic_api_key"
                )
        elif self.provider == Provider.OPENAI:
            if not self.openai_api_key and not self._get_env("OPENAI_API_KEY"):
                raise ConfigValidationError(
                    "OPENAI_API_KEY is required when using the OpenAI provider. "
                    "Set it via FURROW_OPENAI_API_KEY environment variable or .env file.",
                    "openai_api_key"
                )
        elif self.provider == Provider.OLLAMA:
            # Ollama doesn't require an API key, but we can validate the URL is reachable
            pass
        return self

    @staticmethod
    def _get_env(key: str) -> str | None:
        import os
        return os.getenv(key)

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def validate_for_run(self) -> list[str]:
        """Validate settings for a run and return a list of warnings (non-fatal issues)."""
        warnings = []
        
        # Check if workspace exists
        if not self.workspace.exists():
            warnings.append(f"Workspace directory does not exist: {self.workspace}")
        elif not self.workspace.is_dir():
            warnings.append(f"Workspace path is not a directory: {self.workspace}")
        
        # Warn about high token limits
        if self.max_tokens > 8192:
            warnings.append(f"max_tokens ({self.max_tokens}) is very high, may increase costs")
        
        # Warn about low timeouts
        if self.request_timeout < 30:
            warnings.append(f"request_timeout ({self.request_timeout}s) is very low, may cause failures")
        if self.command_timeout < 60:
            warnings.append(f"command_timeout ({self.command_timeout}s) is low, commands may time out")
        if self.test_timeout < 60:
            warnings.append(f"test_timeout ({self.test_timeout}s) is low, tests may time out")
        
        # Warn about Ollama if not running locally
        if self.provider == Provider.OLLAMA and not self.ollama_base_url.startswith(("http://localhost", "http://127.0.0.1")):
            warnings.append(f"Using remote Ollama at {self.ollama_base_url}, ensure it's accessible")
        
        return warnings


settings = Settings()
