from __future__ import annotations

from enum import Enum
from pathlib import Path

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
    # Security: additional denied command patterns (extends built-in DENIED_COMMANDS)
    denied_commands: list[str] = Field(default_factory=list)
    # Security: additional denied program names (extends built-in DENIED_PROGRAMS)
    denied_programs: list[str] = Field(default_factory=list)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return upper

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://")
        return v.rstrip("/")

    @field_validator("workspace", mode="before")
    @classmethod
    def _resolve_workspace(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        return v.resolve()

    @field_validator("state_file", mode="before")
    @classmethod
    def _validate_state_file(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            v = Path(v)
        # Ensure it's a relative path (will be resolved against workspace)
        if v.is_absolute():
            raise ValueError("state_file must be a relative path")
        return v

    @model_validator(mode="after")
    def _validate_provider_settings(self) -> "Settings":
        """Validate provider-specific settings."""
        if self.provider == Provider.ANTHROPIC and not self.anthropic_api_key:
            # Will be checked at runtime when client is created
            pass
        if self.provider == Provider.OPENAI and not self.openai_api_key:
            # Will be checked at runtime when client is created
            pass
        if self.provider == Provider.OLLAMA:
            # Ollama doesn't need an API key
            pass
        return self

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def denied_commands_all(self) -> tuple[str, ...]:
        """Return all denied command patterns (built-in + user-configured)."""
        from furrow.agents.worker import DENIED_COMMANDS as BUILTIN_DENIED_COMMANDS
        return BUILTIN_DENIED_COMMANDS + tuple(self.denied_commands)

    def denied_programs_all(self) -> frozenset[str]:
        """Return all denied program names (built-in + user-configured)."""
        from furrow.agents.worker import DENIED_PROGRAMS as BUILTIN_DENIED_PROGRAMS
        return BUILTIN_DENIED_PROGRAMS | frozenset(self.denied_programs)

    def validate_api_keys(self) -> list[str]:
        """Validate that required API keys are present for the selected provider.
        
        Returns:
            List of warning/error messages. Empty if all required keys are present.
        """
        messages = []
        if self.provider == Provider.ANTHROPIC and not self.anthropic_api_key:
            messages.append(
                "ANTHROPIC_API_KEY not set (required for Anthropic provider). "
                "Set FURROW_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY environment variable."
            )
        if self.provider == Provider.OPENAI and not self.openai_api_key:
            messages.append(
                "OPENAI_API_KEY not set (required for OpenAI provider). "
                "Set FURROW_OPENAI_API_KEY or OPENAI_API_KEY environment variable."
            )
        return messages


settings = Settings()
