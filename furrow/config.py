from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Lifecycle of a single planned task.

    Inherits from ``str`` so existing comparisons such as
    ``task.status == "completed"`` keep working.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str
    # Set by the planner when it judges the goal fully achieved, so the
    # orchestrator loop can terminate cleanly instead of looping forever.
    complete: bool = False


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    # A skipped verification means nothing actually ran (no usable test runner),
    # so it is neither a pass nor a failure. Callers must branch on this flag
    # rather than infer a skip from ``passed`` or from the summary text.
    skipped: bool = False


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
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    # How many times to re-ask the LLM when a response cannot be parsed.
    max_retries: int = 2
    # Seconds before a test command is killed.
    test_timeout: int = 120
    # Explicit test command; when None the tester auto-detects a runner.
    test_command: str | None = None
    # Stop after this many consecutive cycles that make no progress.
    max_idle_cycles: int = 3
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()
