from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
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

    @field_validator("status")
    @classmethod
    def _normalize_status(cls, v: str) -> str:
        if v is None:
            return "pending"
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


class ModelConfig(BaseModel):
    """Configuration for a specific model role."""

    provider: Provider | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    max_tokens: int | None = None
    request_timeout: float | None = None

    def resolve(self, defaults: "ModelConfig") -> "ModelConfig":
        """Resolve this config against defaults, returning a new config with all fields filled."""
        return ModelConfig(
            provider=self.provider or defaults.provider,
            model=self.model or defaults.model,
            api_key=self.api_key or defaults.api_key,
            base_url=self.base_url or defaults.base_url,
            max_tokens=self.max_tokens or defaults.max_tokens,
            request_timeout=self.request_timeout or defaults.request_timeout,
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    # Default model configuration (used as fallback for all roles)
    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_tokens: int = Field(default=4096, ge=1)
    request_timeout: float = Field(default=120.0, gt=0)

    # Role-specific model configurations (optional, override defaults)
    planner_model: str = "claude-3-5-haiku-20241022"
    planner_provider: Provider | None = None
    planner_api_key: str | None = None
    planner_max_tokens: int | None = None
    planner_request_timeout: float | None = None

    worker_model: str = "claude-3-5-sonnet-20241022"
    worker_provider: Provider | None = None
    worker_api_key: str | None = None
    worker_max_tokens: int | None = None
    worker_request_timeout: float | None = None

    tester_model: str = "claude-3-5-sonnet-20241022"
    tester_provider: Provider | None = None
    tester_api_key: str | None = None
    tester_max_tokens: int | None = None
    tester_request_timeout: float | None = None

    # Execution settings
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    max_worker_iterations: int = Field(default=12, ge=1)
    max_retries: int = Field(default=3, ge=0)
    max_task_retries: int = Field(default=1, ge=0)
    command_timeout: float = Field(default=300.0, gt=0)
    test_timeout: float = Field(default=600.0, gt=0)
    state_file: Path = Path(".furrow/state.json")
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def get_model_config(self, role: str) -> ModelConfig:
        """Get the resolved model configuration for a specific role."""
        defaults = ModelConfig(
            provider=self.provider,
            model=self.model,
            api_key=self._get_default_api_key(),
            base_url=self.ollama_base_url if self.provider == Provider.OLLAMA else None,
            max_tokens=self.max_tokens,
            request_timeout=self.request_timeout,
        )

        role = role.lower()
        if role == "planner":
            override = ModelConfig(
                provider=self.planner_provider,
                model=self.planner_model,
                api_key=self.planner_api_key,
                max_tokens=self.planner_max_tokens,
                request_timeout=self.planner_request_timeout,
            )
        elif role == "worker":
            override = ModelConfig(
                provider=self.worker_provider,
                model=self.worker_model,
                api_key=self.worker_api_key,
                max_tokens=self.worker_max_tokens,
                request_timeout=self.worker_request_timeout,
            )
        elif role == "tester":
            override = ModelConfig(
                provider=self.tester_provider,
                model=self.tester_model,
                api_key=self.tester_api_key,
                max_tokens=self.tester_max_tokens,
                request_timeout=self.tester_request_timeout,
            )
        else:
            return defaults

        return override.resolve(defaults)

    def _get_default_api_key(self) -> str | None:
        """Get the default API key based on the provider."""
        if self.provider == Provider.ANTHROPIC:
            return self.anthropic_api_key
        elif self.provider == Provider.OPENAI:
            return self.openai_api_key
        elif self.provider == Provider.OLLAMA:
            return None  # Ollama doesn't need an API key
        return None


settings = Settings()