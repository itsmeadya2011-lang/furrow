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

    @field_validator("description")
    @classmethod
    def _validate_description(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Task description cannot be empty")
        return v.strip()

    @field_validator("files", "dependencies", mode="before")
    @classmethod
    def _ensure_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            return [str(item).strip() for item in v if str(item).strip()]
        return [str(v).strip()] if str(v).strip() else []


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str = ""

    @field_validator("tasks")
    @classmethod
    def _validate_tasks(cls, v: list[TaskModel]) -> list[TaskModel]:
        if len(v) > 10:
            raise ValueError("Maximum 10 tasks per plan allowed")
        # Check for duplicate IDs
        ids = [t.id for t in v]
        if len(ids) != len(set(ids)):
            raise ValueError("Task IDs must be unique")
        return v

    @field_validator("rationale")
    @classmethod
    def _validate_rationale(cls, v: str) -> str:
        return v.strip() if v else ""


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


# Known model prefixes for validation
_KNOWN_MODELS = {
    Provider.ANTHROPIC: ("claude-",),
    Provider.OPENAI: ("gpt-", "o1", "o3", "o4"),
    Provider.OLLAMA: (),  # Ollama models are local, any name is valid
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env", extra="ignore")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = Field(default=5, ge=1, le=20)
    max_cycles: int = Field(default=0, ge=0, le=100)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    max_worker_iterations: int = Field(default=12, ge=1, le=50)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_task_retries: int = Field(default=1, ge=0, le=5)
    request_timeout: float = Field(default=120.0, gt=0, le=600)
    command_timeout: float = Field(default=300.0, gt=0, le=3600)
    test_timeout: float = Field(default=600.0, gt=0, le=7200)
    max_tokens: int = Field(default=4096, ge=1, le=100000)
    state_file: Path = Path(".furrow/state.json")
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.strip().upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def _validate_model_names(self) -> "Settings":
        """Warn if model names don't match expected patterns for the provider."""
        # This is a soft validation - we don't raise, just log a warning would be ideal
        # But we can't log here easily, so we just ensure the fields are non-empty
        for field_name in ("model", "planner_model", "worker_model", "tester_model"):
            value = getattr(self, field_name)
            if not value or not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
        return self

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def validate_api_keys(self) -> list[str]:
        """Validate that required API keys are present for the selected provider.
        Returns a list of missing/invalid key names."""
        missing = []
        if self.provider == Provider.ANTHROPIC:
            if not self.anthropic_api_key and not __import__("os").getenv("ANTHROPIC_API_KEY"):
                missing.append("anthropic_api_key (or ANTHROPIC_API_KEY env var)")
        elif self.provider == Provider.OPENAI:
            if not self.openai_api_key and not __import__("os").getenv("OPENAI_API_KEY"):
                missing.append("openai_api_key (or OPENAI_API_KEY env var)")
        elif self.provider == Provider.OLLAMA:
            # Ollama doesn't require an API key
            pass
        return missing


settings = Settings()
