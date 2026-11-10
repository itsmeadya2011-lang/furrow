"""Configuration and data models for Furrow.

This module defines all the core data models used by Furrow, including task
definitions, plan structures, test results, worker results, token usage,
run results, and application settings.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from furrow.costs import ModelPricing


class Provider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Possible status values for a task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    """A single task in a Furrow plan.

    Attributes:
        id: Unique identifier for the task.
        description: Human-readable description of what the task should accomplish.
        files: List of file paths the task is expected to modify.
        dependencies: List of task IDs that must complete before this task can run.
        status: Current status of the task (normalized to known values).
        result: Summary of what the task accomplished (if completed).
        error: Error message if the task failed.
        attempts: Number of times this task has been attempted.
        started_at: ISO timestamp when the task started.
        finished_at: ISO timestamp when the task finished.
        files_changed: List of file paths that were actually modified.
    """

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

    @field_validator("status")
    @classmethod
    def _normalize_status(cls, v: str) -> str:
        """Normalize status to a known value, defaulting to 'pending'."""
        if v is None:
            return "pending"
        if not isinstance(v, str):
            return "pending"
        known = {"pending", "running", "completed", "failed", "blocked", "skipped"}
        s = v.strip().lower()
        if s in known:
            return s
        return "pending"


class Plan(BaseModel):
    """A plan consisting of multiple tasks with a rationale.

    Attributes:
        tasks: List of tasks to execute.
        rationale: Explanation of the planning decisions.
    """

    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    """Result of running the test suite.

    Attributes:
        passed: Whether the tests passed.
        summary: Human-readable summary of the test run.
        failures: List of individual failure descriptions.
        skipped: Whether the test run was skipped (e.g., no test runner found).
    """

    __test__ = False
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    skipped: bool = False


class WorkerResult(BaseModel):
    """Result of a worker executing a task.

    Attributes:
        summary: Summary of what the worker did.
        success: Whether the worker succeeded.
        files_changed: List of files modified by the worker.
        error: Error message if the worker failed.
        iterations: Number of tool-calling iterations used.
    """

    summary: str = ""
    success: bool = True
    files_changed: list[str] = Field(default_factory=list)
    error: str | None = None
    iterations: int = 0


class Usage(BaseModel):
    """Token usage accumulated over one or more LLM calls.

    Attributes:
        input_tokens: Total input tokens across all calls.
        output_tokens: Total output tokens across all calls.
        calls: Number of LLM calls made.
        cost_usd: Estimated cost in USD.
    """

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
    """Final result of a Furrow run.

    Attributes:
        passed: Whether the tests passed (or were skipped).
        cycles: Number of orchestration cycles completed.
        tasks_completed: Number of tasks that completed successfully.
        tasks_failed: Number of tasks that failed.
        tasks_blocked: Number of tasks that were blocked by failed dependencies.
        tests_run: Number of times the test suite was executed.
        summary: Brief summary of why the run stopped.
        stopped_reason: Detailed reason for stopping.
        skipped: Whether the run was skipped (no test runner detected).
        usage: Accumulated LLM token usage.
        files_changed: List of all files changed during the run.
    """

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
    """Application settings loaded from environment variables.

    All settings use the ``FURROW_`` prefix (e.g., ``FURROW_MODEL``).
    A ``.env`` file is also read if present.

    Attributes:
        provider: LLM provider to use.
        model: Default model for completions/chat.
        planner_model: Model for the planner agent.
        worker_model: Model for the worker agent.
        tester_model: Model for the tester agent.
        anthropic_api_key: Anthropic API key.
        openai_api_key: OpenAI API key.
        ollama_base_url: Ollama OpenAI-compatible base URL.
        max_parallel_tasks: Maximum concurrent worker tasks per wave.
        max_cycles: Maximum cycles to run (0 = unlimited).
        workspace: Working directory for the agent.
        log_level: Logging level used by the CLI.
        max_worker_iterations: Maximum tool-call iterations per worker task.
        max_retries: Retry attempts for transient LLM errors.
        max_task_retries: Times a failed task is retried before marking failed.
        request_timeout: Request timeout in seconds for LLM calls.
        command_timeout: Timeout in seconds for worker commands.
        test_timeout: Timeout in seconds for test commands.
        max_tokens: Max tokens for LLM completions.
        state_file: State file path (relative to workspace).
        model_pricing: User-supplied price overrides for cost estimation.
    """

    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    max_worker_iterations: int = Field(default=12, ge=1)
    max_retries: int = Field(default=3, ge=0)
    max_task_retries: int = Field(default=1, ge=0)
    request_timeout: float = Field(default=120.0, gt=0)
    command_timeout: float = Field(default=300.0, gt=0)
    test_timeout: float = Field(default=600.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    state_file: Path = Path(".furrow/state.json")
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)


settings = Settings()
