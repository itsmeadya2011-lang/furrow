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


# Known model families for validation and pricing fallback
KNOWN_MODEL_FAMILIES = {
    "anthropic": [
        "claude-opus-4",
        "claude-opus-4-1",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "claude-3-7-sonnet",
        "claude-3-5-opus",
        "claude-3-5-sonnet",
        "claude-3-5-haiku",
        "claude-3-opus",
        "claude-3-sonnet",
        "claude-3-haiku",
    ],
    "openai": [
        "gpt-4.1",
        "gpt-4.1-mini",
        "gpt-4.1-nano",
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
        "o1",
        "o1-mini",
        "o3",
        "o3-mini",
        "o4-mini",
    ],
    "ollama": [
        "llama3",
        "llama3.1",
        "llama3.2",
        "mistral",
        "mixtral",
        "codellama",
        "phi3",
        "gemma",
        "gemma2",
    ],
}


def _normalize_model_name(name: str) -> str:
    """Normalize a model name for comparison."""
    return name.strip().lower().replace(".", "-").replace("_", "-")


def _validate_model_for_provider(model: str, provider: Provider) -> str:
    """Validate that a model is appropriate for the given provider.

    This is a soft validation - it warns but doesn't block unknown models
    since new models are released frequently.
    """
    if not model or not model.strip():
        raise ValueError("Model name cannot be empty")

    normalized = _normalize_model_name(model)
    known = KNOWN_MODEL_FAMILIES.get(provider.value, [])

    # Check if the model matches a known family prefix
    for family in known:
        if normalized.startswith(_normalize_model_name(family)):
            return model

    # For ollama, any model name is acceptable since users run custom models
    if provider == Provider.OLLAMA:
        return model

    # For other providers, we allow unknown models but could log a warning
    # In the future, this could be made stricter
    return model


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
    tool_usage: dict[str, dict] = Field(default_factory=dict)
    iteration_summaries: list[dict] = Field(default_factory=list)

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
    tool_usage: dict[str, dict] = Field(default_factory=dict)
    iteration_summaries: list[dict] = Field(default_factory=list)


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
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env", extra="ignore")

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

    @field_validator("model")
    @classmethod
    def _validate_model(cls, v: str, info: Any) -> str:
        # We can't access other fields here easily, so we just do basic validation
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v

    @field_validator("planner_model", "worker_model", "tester_model")
    @classmethod
    def _validate_specialized_models(cls, v: str, info: Any) -> str:
        if not v or not v.strip():
            raise ValueError("Model name cannot be empty")
        return v

    @model_validator(mode="after")
    def _validate_models_for_provider(self) -> "Settings":
        """Validate that all models are appropriate for the selected provider."""
        # Validate main model
        self.model = _validate_model_for_provider(self.model, self.provider)
        # Validate specialized models
        self.planner_model = _validate_model_for_provider(self.planner_model, self.provider)
        self.worker_model = _validate_model_for_provider(self.worker_model, self.provider)
        self.tester_model = _validate_model_for_provider(self.tester_model, self.provider)
        return self

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {sorted(valid)}")
        return upper

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://")
        return v.rstrip("/")

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def validate_api_keys(self) -> list[str]:
        """Validate that required API keys are present for the selected provider.

        Returns a list of warning messages (empty if all keys are present).
        """
        warnings = []
        if self.provider == Provider.ANTHROPIC:
            if not self.anthropic_api_key and not self._env_get("ANTHROPIC_API_KEY"):
                warnings.append("ANTHROPIC_API_KEY is not set (required for Anthropic provider)")
        elif self.provider == Provider.OPENAI:
            if not self.openai_api_key and not self._env_get("OPENAI_API_KEY"):
                warnings.append("OPENAI_API_KEY is not set (required for OpenAI provider)")
        elif self.provider == Provider.OLLAMA:
            # Ollama doesn't require an API key
            pass
        return warnings

    def _env_get(self, key: str) -> str | None:
        import os
        return os.getenv(key)

    def get_model_info(self) -> dict[str, str]:
        """Return a dictionary of model information for display/debugging."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "planner_model": self.planner_model,
            "worker_model": self.worker_model,
            "tester_model": self.tester_model,
        }


settings = Settings()
