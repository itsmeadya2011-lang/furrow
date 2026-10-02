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
    """Lifecycle state of a planned task.

    Subclasses ``str`` so existing ``task.status == "completed"`` comparisons keep working.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    attempts: int = 0
    error: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    """Verdict of one test-suite run.

    ``__test__ = False`` is a pytest opt-out, not a field: pytest collects any imported
    class whose name starts with ``Test``, so without it every module importing this
    model emits a :class:`pytest.PytestCollectionWarning`.  Setting it here keeps the
    public name intact and silences the warning at the source.
    """

    __test__ = False

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    exit_code: int | None = None
    command: str | None = None


class RunResult(BaseModel):
    """Summary of an orchestrator run.

    ``tasks_completed`` counts every task that ended the run in a terminal *done* state
    (``COMPLETED`` or ``SKIPPED``, i.e. finished now or finished in an earlier cycle), so
    it agrees with ``completed`` for a run whose last cycle only skipped settled work.
    ``tasks_failed`` counts ``FAILED`` tasks only.
    """

    goal: str
    cycles: int = 0
    completed: bool = False
    reason: str = ""
    tests_passed: bool = False
    tasks_completed: int = 0
    tasks_failed: int = 0


class Settings(BaseSettings):
    """Runtime configuration, overridable via ``FURROW_``-prefixed env vars or a ``.env`` file.

    ``max_cycles=0`` means unlimited cycles: the orchestrator keeps planning and fixing until the
    goal is reached (or an operator stops it).  Unlimited still terminates on its own -- after
    ``max_task_attempts`` failed attempts a task is abandoned and the run stops with ``no progress``
    once two consecutive cycles complete no task.  ``workspace`` is the sandbox root that bounds every
    file operation performed by workers -- paths escaping it are rejected.
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
    ollama_model: str = "qwen2.5-coder:7b"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    # A task is abandoned (marked FAILED, never dispatched again) once it has been attempted
    # this many times, so a planner that keeps re-planning impossible work cannot loop forever.
    max_task_attempts: int = 3
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    # LLM request behaviour
    temperature: float = 0.2
    max_tokens: int = 4096
    max_retries: int = 3
    retry_base_delay: float = 0.5
    request_timeout: float = 120.0

    # Agent behaviour
    planner_max_tasks: int = 5
    worker_max_turns: int = 12
    test_command_timeout: float = 180.0

    # Workspace scanning
    max_output_chars: int = 20000
    ignore_dirs: list[str] = [
        ".git",
        ".hg",
        ".svn",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        ".tox",
        "dist",
        "build",
        "target",
        ".idea",
        ".vscode",
        ".next",
        ".cache",
        "htmlcov",
        ".eggs",
    ]
    ignore_globs: list[str] = ["*.pyc", "*.pyo", "*.so", "*.lock", "*.log"]
    max_file_list: int = 500

    verbose: bool = False


settings = Settings()
