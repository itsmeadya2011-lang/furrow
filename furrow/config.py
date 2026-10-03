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
    # validate_assignment lets callers keep assigning plain strings (e.g. task.status = "failed");
    # pydantic coerces them to the matching TaskStatus member.
    model_config = ConfigDict(validate_assignment=True)

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    # Not a pytest test class: pytest collects any imported name starting with
    # "Test", and this BaseModel has a generated __init__, which makes pytest
    # emit a PytestCollectionWarning. Do NOT "clean this up" -- the name is
    # public API, so it stays; __test__ = False is how pytest is told to skip it.
    __test__ = False

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class CycleReport(BaseModel):
    cycle: int
    planned: int = 0
    completed: int = 0
    failed: int = 0
    tests_passed: bool | None = None
    summary: str = ""


class RunResult(BaseModel):
    goal: str
    cycles: int = 0
    stopped_reason: str = "goal_complete"  # goal_complete | max_cycles | stopped | error
    tasks: list[TaskModel] = Field(default_factory=list)
    test_results: list[TestResult] = Field(default_factory=list)
    reports: list[CycleReport] = Field(default_factory=list)


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
    ollama_model: str = "llama3"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    max_task_retries: int = 1
    llm_max_retries: int = 3
    llm_max_tokens: int = 4096
    test_command: str | None = None
    test_timeout: int = 120
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    web_host: str = "127.0.0.1"
    web_port: int = 8000


settings = Settings()
