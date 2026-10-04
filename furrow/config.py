"""
Furrow configuration and data models.

This module defines the settings, enums, and Pydantic models used throughout
the Furrow agentic coding system.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"  # Not yet implemented; accepted by config but raises at runtime


class TaskStatus(str, Enum):
    """Status of a task in the execution plan."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskModel(BaseModel):
    """A single task in an execution plan."""

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = Field(default=TaskStatus.PENDING)
    result: str | None = None


class Plan(BaseModel):
    """An execution plan consisting of multiple tasks."""

    tasks: list[TaskModel]
    rationale: str = ""


class TestResult(BaseModel):
    """Result of running the test suite."""

    __test__ = False  # Prevents pytest from collecting this as a test class

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class FileEdit(BaseModel):
    """A single file modification proposed by the worker agent.

    Deliberately permissive: the model is a source of untrusted input, so a malformed
    edit is reported in :attr:`WorkerResult.errors` for that edit alone instead of
    invalidating the whole batch (and the task with it).
    """

    path: str
    action: Literal["write", "edit"] = "write"
    content: str | None = None  # full file text, for action="write"
    search: str | None = None  # unique anchor text, for action="edit"
    replace: str | None = None  # text that replaces the anchor


class WorkerResult(BaseModel):
    """Outcome of one worker task: a summary plus the edits that were applied."""

    __test__ = False  # Prevents pytest from collecting this as a test class

    summary: str
    status: TaskStatus = TaskStatus.COMPLETED
    edits: list[FileEdit] = Field(default_factory=list)
    applied: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    def __str__(self) -> str:
        """Return the summary so string usage of the old ``run() -> str`` API works."""
        return self.summary


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    # Current Anthropic model IDs (2026-10-04): using non-dated aliases that resolve
    # to the latest snapshot. Verified via Anthropic docs (models overview).
    # - Worker: claude-sonnet-5-5 (best speed/intelligence balance for coding)
    # - Planner/Tester: claude-haiku-4-5 (fast, cheap, near-frontier)
    model: str = "claude-sonnet-5-5"
    planner_model: str = "claude-haiku-4-5"
    worker_model: str = "claude-sonnet-5-5"
    tester_model: str = "claude-haiku-4-5"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Maximum concurrent worker tasks per cycle
    max_parallel_tasks: int = 5
    # Maximum orchestrator cycles; 0 means unlimited
    max_cycles: int = 0
    # Root directory the agent is allowed to read and write
    workspace: Annotated[Path, Field(default_factory=Path.cwd)]
    log_level: str = "INFO"
    # LLM client tunables
    llm_timeout: float = 120.0  # seconds, per-request timeout for LLM API calls
    llm_max_retries: int = 3  # retry attempts with exponential backoff
    llm_max_tokens: int = 8192  # max tokens per request (replaces hardcoded 4096)
    test_timeout: int = 120  # seconds before the test-runner subprocess is killed


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


def reload_settings() -> Settings:
    """Clear the settings cache and return a fresh Settings instance."""
    get_settings.cache_clear()
    return get_settings()


settings = get_settings()
