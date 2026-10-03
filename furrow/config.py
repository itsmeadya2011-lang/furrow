from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    # validate_assignment keeps the enum honest when agents assign raw strings,
    # while the ``str`` mixin keeps ``task.status == "completed"`` working.
    model_config = ConfigDict(validate_assignment=True)

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    attempts: int = 0


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str = ""


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    command: str | None = None
    returncode: int | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    ollama_model: str = "llama3.1"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    # max_cycles == 0 means "unbounded" and is bounded by the other guards instead.
    max_cycles: int = 0
    max_parallel_tasks: int = 5
    max_task_attempts: int = 2
    max_consecutive_errors: int = 3
    stagnation_limit: int = 3
    test_command: str | None = None
    test_timeout: int = 120
    max_output_chars: int = 8000
    temperature: float = 0.0
    max_tokens: int = 4096
    state_path: Path | None = None
    verbose: bool = False
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    @property
    def resolved_state_path(self) -> Path:
        """Where :class:`furrow.state.StateStore` persists run state by default."""
        if self.state_path is not None:
            return self.state_path
        return self.workspace / ".furrow" / "state.json"


settings = Settings()