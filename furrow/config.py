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


_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class Settings(BaseSettings):
    """Furrow configuration settings.

    All settings can be overridden via environment variables with the ``FURROW_``
    prefix (e.g., ``FURROW_MAX_CYCLES=10``) or via a ``.env`` file in the
    workspace directory.
    """

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

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Validate log level is a known level."""
        upper = v.upper()
        if upper not in _VALID_LOG_LEVELS:
            raise ValueError(
                f"Invalid log_level: {v!r}. Must be one of: {', '.join(sorted(_VALID_LOG_LEVELS))}"
            )
        return upper

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_base_url(cls, v: str) -> str:
        """Validate ollama_base_url is a valid URL."""
        if not v:
            raise ValueError("ollama_base_url cannot be empty")
        v = v.strip()
        if not (v.startswith("http://") or v.startswith("https://")):
            raise ValueError(f"ollama_base_url must start with http:// or https://, got: {v!r}")
        # Ensure no trailing slash issues
        return v.rstrip("/")

    @field_validator("state_file")
    @classmethod
    def _validate_state_file(cls, v: Path) -> Path:
        """Validate state_file is a relative path (for security)."""
        if v.is_absolute():
            raise ValueError(
                f"state_file must be a relative path, got absolute path: {v!r}. "
                "Use a relative path like '.furrow/state.json'"
            )
        return v

    @field_validator("workspace", mode="before")
    @classmethod
    def _resolve_workspace(cls, v: Any) -> Path:
        """Resolve workspace to an absolute path."""
        if isinstance(v, str):
            v = Path(v)
        return v.resolve()

    @model_validator(mode="after")
    def _validate_provider_keys(self) -> "Settings":
        """Validate that required API keys are present for the selected provider."""
        if self.provider == Provider.ANTHROPIC and not self.anthropic_api_key:
            # Allow missing key if it can be read from environment later
            pass
        if self.provider == Provider.OPENAI and not self.openai_api_key:
            pass
        return self

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def validate_for_run(self) -> list[str]:
        """Validate settings for a run and return a list of warnings/errors.

        Returns:
            List of validation messages. Empty list means all checks passed.
            Messages starting with "ERROR:" indicate configuration that will
            cause the run to fail; others are warnings.
        """
        issues: list[str] = []

        # Check API keys for the selected provider
        if self.provider == Provider.ANTHROPIC:
            if not self.anthropic_api_key and not __import__("os").getenv("ANTHROPIC_API_KEY"):
                issues.append(
                    "ERROR: ANTHROPIC_API_KEY is not set (required for anthropic provider). "
                    "Set FURROW_ANTHROPIC_API_KEY or the ANTHROPIC_API_KEY environment variable."
                )
        elif self.provider == Provider.OPENAI:
            if not self.openai_api_key and not __import__("os").getenv("OPENAI_API_KEY"):
                issues.append(
                    "ERROR: OPENAI_API_KEY is not set (required for openai provider). "
                    "Set FURROW_OPENAI_API_KEY or the OPENAI_API_KEY environment variable."
                )
        elif self.provider == Provider.OLLAMA:
            # Ollama doesn't require an API key, but we can check if the URL is reachable
            pass

        # Warn about potentially problematic settings
        if self.max_cycles == 0:
            issues.append(
                "WARNING: max_cycles is 0 (unlimited). The run may continue indefinitely "
                "if the no-progress guard doesn't trigger."
            )

        if self.max_worker_iterations > 50:
            issues.append(
                f"WARNING: max_worker_iterations is {self.max_worker_iterations} (high). "
                "This may lead to long-running worker tasks."
            )

        if self.request_timeout > 300:
            issues.append(
                f"WARNING: request_timeout is {self.request_timeout}s (high). "
                "Consider a lower value for faster failure detection."
            )

        if self.command_timeout < 60:
            issues.append(
                f"WARNING: command_timeout is {self.command_timeout}s (low). "
                "Some commands may need more time to complete."
            )

        if self.test_timeout < 120:
            issues.append(
                f"WARNING: test_timeout is {self.test_timeout}s (low). "
                "Test suites may need more time to complete."
            )

        return issues


settings = Settings()