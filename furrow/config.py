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
    GEMINI = "gemini"


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
    gemini_api_key: str | None = None
    max_parallel_tasks: int = Field(default=5, ge=1, le=50)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    max_worker_iterations: int = Field(default=12, ge=1, le=100)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_task_retries: int = Field(default=1, ge=0, le=10)
    request_timeout: float = Field(default=120.0, gt=0, le=600)
    command_timeout: float = Field(default=300.0, gt=0, le=3600)
    test_timeout: float = Field(default=600.0, gt=0, le=7200)
    max_tokens: int = Field(default=4096, ge=1, le=100000)
    state_file: Path = Path(".furrow/state.json")
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)
    cost_budget_usd: float = Field(default=0.0, ge=0, le=10000)

    @field_validator("log_level", mode="before")
    @classmethod
    def _validate_log_level(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "INFO"
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.strip().upper()
        if upper in valid_levels:
            return upper
        return "INFO"

    @field_validator("ollama_base_url", mode="before")
    @classmethod
    def _validate_ollama_url(cls, v: Any) -> str:
        if not isinstance(v, str):
            return "http://localhost:11434"
        url = v.strip()
        if not url:
            return "http://localhost:11434"
        # Ensure it has a scheme
        if not url.startswith(("http://", "https://")):
            url = "http://" + url
        return url.rstrip("/")

    @model_validator(mode="after")
    def _validate_provider_keys(self) -> Settings:
        """Validate that required API keys are present for the selected provider."""
        # This is a warning-level validation - we don't raise here because
        # the keys might be provided via environment variables at runtime
        return self

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def validate_keys(self) -> list[str]:
        """Check for missing API keys and return a list of warnings."""
        warnings = []
        if self.provider == Provider.ANTHROPIC and not (
            self.anthropic_api_key or self._get_env("ANTHROPIC_API_KEY")
        ):
            warnings.append("ANTHROPIC_API_KEY not set (required for Anthropic provider)")
        elif self.provider == Provider.OPENAI and not (
            self.openai_api_key or self._get_env("OPENAI_API_KEY")
        ):
            warnings.append("OPENAI_API_KEY not set (required for OpenAI provider)")
        elif self.provider == Provider.GEMINI and not (
            self.gemini_api_key
            or self._get_env("GEMINI_API_KEY")
            or self._get_env("GOOGLE_API_KEY")
        ):
            warnings.append(
                "GEMINI_API_KEY or GOOGLE_API_KEY not set (required for Gemini provider)"
            )
        return warnings

    def _get_env(self, key: str) -> str | None:
        import os
        return os.getenv(key)

    @classmethod
    def generate_schema_doc(cls) -> str:
        """Generate a markdown documentation of all configuration options."""
        lines = [
            "# Furrow Configuration Reference",
            "",
            "All settings can be configured via environment variables with the `FURROW_` prefix",
            "or in a `.env` file in the working directory.",
            "",
            "## Provider Settings",
            "",
            "| Setting | Type | Default | Description |",
            "|---------|------|---------|-------------|",
            "| `provider` | enum | `anthropic` | LLM provider: "
            "`anthropic`, `openai`, `ollama`, `gemini` |",
            "| `model` | string | `claude-sonnet-4-20250514` | Default model "
            "for general requests |",
            "| `planner_model` | string | `claude-3-5-haiku-20241022` | Model for planning tasks |",
            "| `worker_model` | string | `claude-3-5-sonnet-20241022` | Model for worker tasks |",
            "| `tester_model` | string | `claude-3-5-sonnet-20241022` | Model for "
            "test verification |",
            "| `anthropic_api_key` | string | - | Anthropic API key |",
            "| `openai_api_key` | string | - | OpenAI API key |",
            "| `gemini_api_key` | string | - | Google Gemini API key |",
            "| `ollama_base_url` | string | `http://localhost:11434` | Ollama server base URL |",
            "",
            "## Execution Settings",
            "",
            "| Setting | Type | Default | Range | Description |",
            "|---------|------|---------|-------|-------------|",
            "| `max_parallel_tasks` | integer | 5 | 1-50 | Maximum concurrent worker tasks |",
            "| `max_cycles` | integer | 0 | 0+ | Maximum planning cycles (0 = unlimited) |",
            "| `max_worker_iterations` | integer | 12 | 1-100 | Max tool "
            "iterations per worker task |",
            "| `max_retries` | integer | 3 | 0-10 | Max retries for LLM API calls |",
            "| `max_task_retries` | integer | 1 | 0-10 | Max retries for failed tasks |",
            "| `request_timeout` | float | 120.0 | 0-600 | Timeout for "
            "LLM API requests (seconds) |",
            "| `command_timeout` | float | 300.0 | 0-3600 | Timeout for shell commands (seconds) |",
            "| `test_timeout` | float | 600.0 | 0-7200 | Timeout for test suite (seconds) |",
            "| `max_tokens` | integer | 4096 | 1-100000 | Max tokens per LLM response |",
            "| `cost_budget_usd` | float | 0.0 | 0-10000 | Cost budget in USD (0 = unlimited) |",
            "",
            "## Workspace Settings",
            "",
            "| Setting | Type | Default | Description |",
            "|---------|------|---------|-------------|",
            "| `workspace` | path | current directory | Project workspace directory |",
            "| `state_file` | path | `.furrow/state.json` | Relative path for run state |",
            "| `log_level` | enum | `INFO` | Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |",
            "",
            "## Cost Tracking",
            "",
            "| Setting | Type | Default | Description |",
            "|---------|------|---------|-------------|",
            "| `model_pricing` | object | {} | Custom model pricing overrides (JSON) |",
        ]
        return "\n".join(lines)


settings = Settings()
