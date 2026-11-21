from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from furrow.costs import ModelPricing


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    GEMINI = "gemini"
    AZURE = "azure"
    VERTEX = "vertex"
    BEDROCK = "bedrock"
    GROQ = "groq"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    XAI = "xai"
    OPENROUTER = "openrouter"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""
    max_attempts: int = Field(default=3, ge=1, description="Maximum number of retry attempts")
    base_delay: float = Field(default=1.0, gt=0, description="Base delay in seconds for exponential backoff")
    max_delay: float = Field(default=20.0, gt=0, description="Maximum delay in seconds between retries")
    exponential_base: float = Field(default=2.0, gt=1, description="Base for exponential backoff")
    retryable_status_codes: set[int] = Field(
        default={408, 409, 425, 429, 500, 502, 503, 504, 529},
        description="HTTP status codes that should trigger a retry"
    )


class ProviderConfig(BaseModel):
    """Provider-specific configuration."""
    api_key: str | None = None
    base_url: str | None = None
    timeout: float = Field(default=120.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    retry_policy: RetryPolicy = Field(default_factory=RetryPolicy)
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_params: dict[str, Any] = Field(default_factory=dict)


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
    
    # Provider-specific configurations
    anthropic: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None))
    openai: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None))
    ollama: ProviderConfig = Field(default_factory=lambda: ProviderConfig(base_url="http://localhost:11434"))
    gemini: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url="https://generativelanguage.googleapis.com/v1beta"))
    azure: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url=None))
    vertex: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url=None))
    bedrock: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url=None))
    groq: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url="https://api.groq.com/openai/v1"))
    deepseek: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url="https://api.deepseek.com/v1"))
    mistral: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url="https://api.mistral.ai/v1"))
    xai: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url="https://api.x.ai/v1"))
    openrouter: ProviderConfig = Field(default_factory=lambda: ProviderConfig(api_key=None, base_url="https://openrouter.ai/api/v1"))
    
    # Legacy fields for backward compatibility
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    log_json: bool = Field(default=False, description="Output logs in JSON format")
    log_timestamp: bool = Field(default=True, description="Include timestamps in log output")
    max_worker_iterations: int = Field(default=12, ge=1)
    max_retries: int = Field(default=3, ge=0)
    max_task_retries: int = Field(default=1, ge=0)
    request_timeout: float = Field(default=120.0, gt=0)
    command_timeout: float = Field(default=300.0, gt=0)
    test_timeout: float = Field(default=600.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    state_file: Path = Path(".furrow/state.json")
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)
    
    # Advanced options
    enable_cost_tracking: bool = Field(default=True, description="Track and report LLM costs")
    enable_telemetry: bool = Field(default=False, description="Enable anonymous usage telemetry")
    planner_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for planner model")
    worker_temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Temperature for worker model")
    tester_temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Temperature for tester model")

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)

    def get_provider_config(self, provider: Provider) -> ProviderConfig:
        """Get the configuration for a specific provider."""
        config_map = {
            Provider.ANTHROPIC: self.anthropic,
            Provider.OPENAI: self.openai,
            Provider.OLLAMA: self.ollama,
            Provider.GEMINI: self.gemini,
            Provider.AZURE: self.azure,
            Provider.VERTEX: self.vertex,
            Provider.BEDROCK: self.bedrock,
            Provider.GROQ: self.groq,
            Provider.DEEPSEEK: self.deepseek,
            Provider.MISTRAL: self.mistral,
            Provider.XAI: self.xai,
            Provider.OPENROUTER: self.openrouter,
        }
        config = config_map.get(provider, ProviderConfig())
        
        # Apply legacy fields for backward compatibility
        if provider == Provider.ANTHROPIC and self.anthropic_api_key:
            config.api_key = self.anthropic_api_key
        elif provider == Provider.OPENAI and self.openai_api_key:
            config.api_key = self.openai_api_key
        elif provider == Provider.OLLAMA and self.ollama_base_url:
            config.base_url = self.ollama_base_url
            
        return config


settings = Settings()
