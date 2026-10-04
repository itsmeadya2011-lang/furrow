from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator
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
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    attempts: int = 0
    error: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str

    @property
    def is_empty(self) -> bool:
        return not self.tasks

    def task_map(self) -> dict[str, TaskModel]:
        return {task.id: task for task in self.tasks}

    def duplicate_ids(self) -> list[str]:
        seen: set[str] = set()
        dupes: list[str] = []
        for task in self.tasks:
            if task.id in seen:
                dupes.append(task.id)
            else:
                seen.add(task.id)
        return dupes


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


PROVIDER_MODELS: dict[Provider, dict[str, str]] = {
    Provider.ANTHROPIC: {
        "model": "claude-sonnet-4-20250514",
        "planner_model": "claude-3-5-haiku-20241022",
        "worker_model": "claude-3-5-sonnet-20241022",
        "tester_model": "claude-3-5-sonnet-20241022",
    },
    Provider.OPENAI: {
        "model": "gpt-4o",
        "planner_model": "gpt-4o-mini",
        "worker_model": "gpt-4o",
        "tester_model": "gpt-4o-mini",
    },
    Provider.OLLAMA: {
        "model": "qwen2.5-coder:14b",
        "planner_model": "qwen2.5-coder:7b",
        "worker_model": "qwen2.5-coder:14b",
        "tester_model": "qwen2.5-coder:7b",
    },
}


#: Bare, unprefixed credential variables. The ``FURROW_``-prefixed form is
#: already consumed by the corresponding ``Settings`` field, but ``furrow.llm``
#: also falls back to these when the field is empty, so a runtime check has to
#: read them too -- otherwise it warns about a key the client can actually find
#: and ``furrow config`` disagrees with what ``furrow start`` accepts.
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"


def _resolve_credential(field_value: str | None, env_name: str) -> str | None:
    """The API key actually reachable for a provider, or ``None``.

    Mirrors the ``self._settings.<field> or os.getenv(...)`` fallback chain in
    :attr:`furrow.llm.LLMClient.anthropic` and ``.openai``, so this report and
    the client that does the work can never disagree about whether a credential
    exists. Advisory only: an empty result means "warn", never "fail".
    """
    return field_value or os.environ.get(env_name)


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
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    test_command: str | None = None
    test_timeout: int = Field(default=120, ge=1)
    worker_max_iterations: int = Field(default=12, ge=1)
    allow_shell: bool = False
    shell_timeout: int = Field(default=60, ge=1)
    log_json: bool = False

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> Settings:
        defaults = PROVIDER_MODELS.get(self.provider, {})
        for field_name, mapped_default in defaults.items():
            if field_name not in self.model_fields_set:
                setattr(self, field_name, mapped_default)
        return self

    def model_for(self, role: str) -> str:
        role_map = {
            "planner": self.planner_model,
            "worker": self.worker_model,
            "tester": self.tester_model,
        }
        return role_map.get(role, self.model)

    def validate_runtime(self) -> list[str]:
        """Advisory report of the configuration problems worth warning about.

        Never raises and never blocks: a warning is a note, not a refusal. The
        credential checks consult the same fallbacks the LLM client does, so a
        key supplied only through the unprefixed environment variable counts as
        configured.
        """
        warnings: list[str] = []
        if (
            self.provider == Provider.ANTHROPIC
            and not _resolve_credential(self.anthropic_api_key, ANTHROPIC_KEY_ENV)
        ):
            warnings.append("provider is anthropic but no API key configured")
        if (
            self.provider == Provider.OPENAI
            and not _resolve_credential(self.openai_api_key, OPENAI_KEY_ENV)
        ):
            warnings.append("provider is openai but no API key configured")
        if not self.workspace.exists():
            warnings.append(f"workspace does not exist: {self.workspace}")
        if self.max_parallel_tasks > 20:
            warnings.append(f"max_parallel_tasks ({self.max_parallel_tasks}) > 20")
        if self.allow_shell:
            warnings.append("allow_shell is enabled")
        return warnings


# Module-level singleton kept for backwards compatibility.
# Import-time failure is possible if required env vars or config are missing.
settings = Settings()
