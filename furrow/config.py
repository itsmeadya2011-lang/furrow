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


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    __test__ = False
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class WorkerResult(BaseModel):
    summary: str = ""
    success: bool = True
    files_changed: list[str] = Field(default_factory=list)
    error: str | None = None
    iterations: int = 0


class RunResult(BaseModel):
    passed: bool = False
    cycles: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    summary: str = ""
    stopped_reason: str = ""


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
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    max_worker_iterations: int = 12
    max_retries: int = 3
    max_task_retries: int = 1
    request_timeout: float = 120.0
    max_tokens: int = 4096
    max_output_chars: int = 20000
    worker_command_timeout: float = 120.0
    state_file: Path = Path(".furrow/state.json")

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)


settings = Settings()
