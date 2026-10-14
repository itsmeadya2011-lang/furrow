from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    cycle: int = 0


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class LintResult(BaseModel):
    passed: bool
    summary: str
    issues: list[str] = Field(default_factory=list)


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
    max_cycles: int = 10
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    test_commands: list[list[str]] = Field(default_factory=lambda: [
        ["pytest", "-q"],
        ["python", "-m", "pytest", "-q"],
        ["npm", "test", "--", "--silent"],
        ["pnpm", "test", "--", "--silent"],
        ["yarn", "test", "--silent"],
        ["cargo", "test", "-q"],
        ["go", "test", "./..."],
    ])
    lint_commands: list[list[str]] = Field(default_factory=lambda: [
        ["ruff", "check", "."],
        ["mypy", "."],
        ["npm", "run", "lint"],
        ["pnpm", "run", "lint"],
        ["yarn", "lint"],
    ])
    type_check_commands: list[list[str]] = Field(default_factory=lambda: [
        ["mypy", "."],
        ["tsc", "--noEmit"],
    ])


settings = Settings()
