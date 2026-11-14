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
    GOOGLE = "google"
    AZURE = "azure"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    MISTRAL = "mistral"
    XAI = "xai"


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


class WebSettings(BaseModel):
    """Web server configuration."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    enable_cors: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    websocket_ping_interval: float = Field(default=20.0, gt=0)
    websocket_ping_timeout: float = Field(default=10.0, gt=0)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FURROW_",
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
    )

    # Provider and model settings
    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"

    # API keys
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    google_api_key: str | None = None
    azure_api_key: str | None = None
    azure_api_base: str | None = None
    azure_api_version: str | None = None
    deepseek_api_key: str | None = None
    groq_api_key: str | None = None
    mistral_api_key: str | None = None
    xai_api_key: str | None = None

    # Execution settings
    max_parallel_tasks: int = Field(default=5, ge=1, le=50)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    # Agent settings
    max_worker_iterations: int = Field(default=12, ge=1, le=100)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_task_retries: int = Field(default=1, ge=0, le=10)

    # Timeout settings
    request_timeout: float = Field(default=120.0, gt=0, le=3600)
    command_timeout: float = Field(default=300.0, gt=0, le=3600)
    test_timeout: float = Field(default=600.0, gt=0, le=7200)

    # Token limits
    max_tokens: int = Field(default=4096, ge=1, le=100000)

    # State persistence
    state_file: Path = Path(".furrow/state.json")

    # Cost tracking
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    # Web server settings
    web: WebSettings = Field(default_factory=WebSettings)

    # Custom denied commands/programs for security
    denied_commands: list[str] = Field(default_factory=list)
    denied_programs: list[str] = Field(default_factory=list)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper

    @field_validator("workspace", mode="before")
    @classmethod
    def _validate_workspace(cls, v: str | Path) -> Path:
        path = Path(v).expanduser().resolve()
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        return path

    @field_validator("state_file", mode="before")
    @classmethod
    def _validate_state_file(cls, v: str | Path) -> Path:
        return Path(v)

    @model_validator(mode="after")
    def _validate_provider_keys(self) -> Settings:
        """Validate that required API keys are present for the selected provider."""
        provider = self.provider
        if provider == Provider.ANTHROPIC and not self.anthropic_api_key:
            import os
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
        elif provider == Provider.OPENAI and not self.openai_api_key:
            import os
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        elif provider == Provider.GOOGLE and not self.google_api_key:
            import os
            if not os.getenv("GOOGLE_API_KEY"):
                raise ValueError("GOOGLE_API_KEY is required for Google provider")
        elif provider == Provider.AZURE and not self.azure_api_key:
            import os
            if not os.getenv("AZURE_API_KEY"):
                raise ValueError("AZURE_API_KEY is required for Azure provider")
        elif provider == Provider.DEEPSEEK and not self.deepseek_api_key:
            import os
            if not os.getenv("DEEPSEEK_API_KEY"):
                raise ValueError("DEEPSEEK_API_KEY is required for DeepSeek provider")
        elif provider == Provider.GROQ and not self.groq_api_key:
            import os
            if not os.getenv("GROQ_API_KEY"):
                raise ValueError("GROQ_API_KEY is required for Groq provider")
        elif provider == Provider.MISTRAL and not self.mistral_api_key:
            import os
            if not os.getenv("MISTRAL_API_KEY"):
                raise ValueError("MISTRAL_API_KEY is required for Mistral provider")
        elif provider == Provider.XAI and not self.xai_api_key:
            import os
            if not os.getenv("XAI_API_KEY"):
                raise ValueError("XAI_API_KEY is required for xAI provider")
        return self

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def get_denied_commands(self) -> tuple[str, ...]:
        """Get the combined list of denied commands (built-in + custom)."""
        from furrow.security.command_validator import DENIED_COMMANDS
        return DENIED_COMMANDS + tuple(self.denied_commands)

    def get_denied_programs(self) -> frozenset[str]:
        """Get the combined set of denied programs (built-in + custom)."""
        from furrow.security.command_validator import DENIED_PROGRAMS
        return DENIED_PROGRAMS | frozenset(self.denied_programs)


settings = Settings()