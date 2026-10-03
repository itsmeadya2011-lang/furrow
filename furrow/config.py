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
    """Lifecycle states for a single task.

    A ``str`` enum so ``task.status == "completed"`` keeps working for legacy
    callers while typos ("complete", "done") now fail validation instead of
    silently corrupting run state.
    """

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
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    error: str | None = None
    attempts: int = 0

    def is_terminal(self) -> bool:
        """True when the task will not run again without intervention."""
        return self.status in _TERMINAL_STATUSES

    def is_done(self) -> bool:
        """True only when the task completed successfully."""
        return self.status is TaskStatus.COMPLETED


_TERMINAL_STATUSES: frozenset[TaskStatus] = frozenset(
    {
        TaskStatus.COMPLETED,
        TaskStatus.FAILED,
        TaskStatus.BLOCKED,
        TaskStatus.SKIPPED,
    }
)


class Plan(BaseModel):
    """A planner response.

    Both fields are tolerant of partial LLM output: a response that omits
    ``rationale`` or carries no tasks validates as an empty plan rather than
    raising and killing the cycle.
    """

    tasks: list[TaskModel] = Field(default_factory=list)
    rationale: str = ""


class TestResult(BaseModel):
    """The verdict for one test cycle.

    ``__test__ = False`` tells pytest not to collect this class purely because
    its name starts with ``Test``. Pydantic ignores dunder class attributes, so
    it is not a model field and never appears in serialised output.
    """

    __test__ = False

    passed: bool
    summary: str = ""
    failures: list[str] = Field(default_factory=list)


#: Canonical default for :attr:`Settings.max_tokens`, overridable with
#: ``FURROW_MAX_TOKENS``. It lives here -- next to the field it defaults -- so
#: callers that want the default without building a ``Settings`` cannot drift
#: away from it; ``furrow.llm`` re-exports it under its original name.
MAX_TOKENS_DEFAULT: int = 8192


class Settings(BaseSettings):
    """Runtime configuration, read from ``FURROW_``-prefixed env vars / ``.env``.

    Loop control: ``FURROW_MAX_CYCLES``, ``FURROW_MAX_PARALLEL_TASKS``,
    ``FURROW_MAX_CONSECUTIVE_FAILURES``, ``FURROW_MAX_TASK_ATTEMPTS``.

    Testing: ``FURROW_TEST_COMMAND``, ``FURROW_TEST_TIMEOUT``.

    LLM: ``FURROW_MODEL``, ``FURROW_MAX_TOKENS`` and the per-role
    ``FURROW_PLANNER_MODEL`` / ``FURROW_WORKER_MODEL`` / ``FURROW_TESTER_MODEL``.

    Persistence: ``FURROW_STATE_FILE`` (defaults to
    ``<workspace>/.furrow/state.json``, resolved at call time).
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
    max_tokens: int = MAX_TOKENS_DEFAULT
    max_parallel_tasks: int = 5
    max_cycles: int = 0  # 0 means unlimited
    max_consecutive_failures: int = 3
    max_task_attempts: int = 2
    test_command: str | None = None
    test_timeout: int = 300
    state_file: Path | None = None
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()
