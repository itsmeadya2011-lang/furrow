from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskModel(BaseModel):
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
    __test__ = False

    passed: bool
    summary: str = ""
    failures: list[str] = Field(default_factory=list)
    commands: list[str] = Field(default_factory=list)


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

    temperature: float = 0.0
    max_tokens: int = 8192
    max_retries: int = 3
    retry_base_delay: float = 1.0
    request_timeout: float = 120.0

    test_command: str | None = None
    test_timeout: int = 300
    lint_command: str | None = None
    typecheck_command: str | None = None
    exclude_dirs: list[str] = Field(
        default_factory=lambda: [
            ".git",
            ".venv",
            "venv",
            "node_modules",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            "dist",
            "build",
            ".eggs",
            "*.egg-info",
        ]
    )

    web_host: str = "127.0.0.1"
    web_port: int = 8000
    state_file: Path | None = None
    max_tool_iterations: int = 12
    max_stalled_cycles: int = 3
    max_event_history: int = 500

    def api_key_for(self, provider: Provider | None = None) -> str | None:
        """Return the API key configured for `provider` (defaults to the active provider)."""
        target = provider or self.provider
        if target == Provider.ANTHROPIC:
            return self.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        if target == Provider.OPENAI:
            return self.openai_api_key or os.getenv("OPENAI_API_KEY")
        return None

    def resolved_workspace(self) -> Path:
        """Return `workspace` as an absolute path, falling back to the cwd if it is gone."""
        if not self.workspace.exists():
            return Path.cwd()
        return self.workspace.resolve()


settings = Settings()


def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return settings


def reload_settings(**overrides: Any) -> Settings:
    """Return a NEW Settings built from the current env plus overrides.

    The process-wide singleton is never mutated.
    """
    return Settings(**overrides)
