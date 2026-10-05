"""Configuration models for furrow.

Every :class:`Settings` field is populated from the environment (or a ``.env``
file) using the ``FURROW_`` prefix, for example ``FURROW_PROVIDER=ollama`` or
``FURROW_LLM_MAX_RETRIES=5``. The unprefixed ``ANTHROPIC_API_KEY`` and
``OPENAI_API_KEY`` variables are still honoured as a fallback by
:mod:`furrow.llm` because the README documents them.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Pinned model snapshots. Check these against the models your account can
# actually reach before bumping them; planning uses the cheapest capable model.
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_PLANNER_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_WORKER_MODEL = "claude-sonnet-5"
DEFAULT_TESTER_MODEL = "claude-sonnet-5"
DEFAULT_OLLAMA_MODEL = "llama3.1"


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    # Not a pytest test class: pytest collects any imported name starting with
    # "Test", and this model has a generated ``__init__``. Opt out explicitly so
    # importing it from a test module does not raise a collection warning.
    __test__ = False

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Runtime configuration for the furrow agent loop."""

    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Field(
        default=Provider.ANTHROPIC,
        description="LLM provider to call: anthropic, openai or ollama.",
    )
    model: str = Field(
        default=DEFAULT_MODEL,
        description="Default model ID used when a caller does not request one.",
    )
    planner_model: str = Field(
        default=DEFAULT_PLANNER_MODEL,
        description="Model ID used to turn a goal into a task plan.",
    )
    worker_model: str = Field(
        default=DEFAULT_WORKER_MODEL,
        description="Model ID used to implement a single task.",
    )
    tester_model: str = Field(
        default=DEFAULT_TESTER_MODEL,
        description="Model ID used to review the workspace and judge the tests.",
    )
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key; falls back to the ANTHROPIC_API_KEY env var.",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key; falls back to the OPENAI_API_KEY env var.",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL of the Ollama server (no /api path suffix).",
    )
    ollama_model: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        description="Ollama model tag, used when the provider is ollama.",
    )
    max_parallel_tasks: int = Field(
        default=5,
        description="Maximum number of worker tasks executed concurrently.",
    )
    max_cycles: int = Field(
        default=0,
        description="Maximum number of plan/execute cycles; 0 means unlimited.",
    )
    max_consecutive_failures: int = Field(
        default=3,
        description="Circuit breaker: stop after this many failing cycles in a row; "
        "0 or less disables it.",
    )
    max_tokens: int = Field(
        default=4096,
        description="Maximum tokens requested per LLM completion.",
    )
    llm_timeout_seconds: float = Field(
        default=120.0,
        description="Per-request timeout, in seconds, for every LLM call.",
    )
    llm_max_retries: int = Field(
        default=3,
        description="Attempts per LLM call, including the first one; 1 disables retries.",
    )
    workspace: Path = Field(
        default_factory=Path.cwd,
        description="Root directory the agents read from and write to.",
    )
    log_level: str = Field(
        default="INFO",
        description="Log level applied by the CLI and web entry points.",
    )
    host: str = Field(
        default="127.0.0.1",
        description="Interface the web server binds to.",
    )
    port: int = Field(
        default=8000,
        description="Port the web server listens on.",
    )


settings = Settings()
