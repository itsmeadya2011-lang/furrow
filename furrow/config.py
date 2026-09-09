"""Configuration models and settings for Furrow.

This module defines the core configuration classes used throughout the Furrow
agent system, including provider settings, task models, plan structures,
and test result models.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    """Supported LLM providers for Furrow agents."""
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Status values for task execution."""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskModel(BaseModel):
    """Represents a single task in an execution plan.
    
    Attributes:
        id: Unique identifier for the task.
        description: Human-readable description of the task.
        files: List of file paths relevant to this task.
        dependencies: List of task IDs that must complete before this task.
        status: Current execution status of the task.
        result: Optional result or output from task execution.
    """
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None


class Plan(BaseModel):
    """Execution plan consisting of multiple tasks with a rationale.
    
    Attributes:
        tasks: Ordered list of tasks to execute.
        rationale: Explanation of the planning decisions.
    """
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    """Result of running tests against implemented changes.
    
    Attributes:
        passed: Whether all tests passed.
        summary: Human-readable summary of test results.
        failures: List of failure messages if any tests failed.
    """
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Global settings for the Furrow agent system.
    
    Configures LLM providers, model selection, API keys, execution parameters,
    workspace location, and logging level. All settings can be overridden via
    environment variables prefixed with FURROW_ or a .env file.
    """
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()
