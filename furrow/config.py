from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

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


def _validate_log_level(v: str) -> str:
    """Validate that the log level is a standard logging level."""
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    upper = v.upper()
    if upper not in valid_levels:
        raise ValueError(f"log_level must be one of {sorted(valid_levels)}")
    return upper


def _validate_model_name(v: str) -> str:
    """Validate that the model name is not empty."""
    if not v or not v.strip():
        raise ValueError("model name cannot be empty")
    return v.strip()


def _validate_ollama_url(v: str) -> str:
    """Validate that the Ollama URL is a valid HTTP/HTTPS URL."""
    if not v:
        raise ValueError("ollama_base_url cannot be empty")
    v = v.strip()
    if not (v.startswith("http://") or v.startswith("https://")):
        raise ValueError("ollama_base_url must start with http:// or https://")
    return v.rstrip("/")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: Annotated[
        str, Field(default="claude-sonnet-4-20250514")
    ] = "claude-sonnet-4-20250514"
    planner_model: Annotated[
        str, Field(default="claude-3-5-haiku-20241022")
    ] = "claude-3-5-haiku-20241022"
    worker_model: Annotated[
        str, Field(default="claude-3-5-sonnet-20241022")
    ] = "claude-3-5-sonnet-20241022"
    tester_model: Annotated[
        str, Field(default="claude-3-5-sonnet-20241022")
    ] = "claude-3-5-sonnet-20241022"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: Annotated[
        str, Field(default="http://localhost:11434")
    ] = "http://localhost:11434"
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: Annotated[
        str, Field(default="INFO")
    ] = "INFO"
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
    def _validate_log_level(cls, v: str) -> str:
        return _validate_log_level(v)

    @field_validator("model", "planner_model", "worker_model", "tester_model", mode="before")
    @classmethod
    def _validate_model_names(cls, v: str) -> str:
        return _validate_model_name(v)

    @field_validator("ollama_base_url", mode="before")
    @classmethod
    def _validate_ollama_base_url(cls, v: str) -> str:
        return _validate_ollama_url(v)

    @field_validator("anthropic_api_key", "openai_api_key", mode="before")
    @classmethod
    def _validate_api_keys(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            return None
        return v

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def validate_api_keys(self) -> list[str]:
        """Validate that required API keys are present for the selected provider.
        
        Returns a list of missing API key names (empty if all present).
        """
        missing = []
        if self.provider == Provider.ANTHROPIC and not self.anthropic_api_key:
            missing.append("FURROW_ANTHROPIC_API_KEY")
        elif self.provider == Provider.OPENAI and not self.openai_api_key:
            missing.append("FURROW_OPENAI_API_KEY")
        elif self.provider == Provider.OLLAMA:
            # Ollama doesn't require an API key, but the base URL must be valid
            pass
        return missing


settings = Settings()
