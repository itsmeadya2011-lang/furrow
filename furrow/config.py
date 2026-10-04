from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    # No Ollama member: furrow.llm has no Ollama completion backend, so offering it here
    # only moved a guaranteed "Unsupported provider" ValueError from startup to the first
    # request. Omitting it turns FURROW_PROVIDER=ollama into an immediate, clear error.


TASK_STATUSES = frozenset({"pending", "running", "completed", "failed"})


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("task id must not be blank")
        return value

    @field_validator("status")
    @classmethod
    def _known_status(cls, value: str) -> str:
        if value not in TASK_STATUSES:
            raise ValueError(
                f"unknown task status {value!r}; expected one of {sorted(TASK_STATUSES)}"
            )
        return value


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str

    @model_validator(mode="after")
    def _self_consistent_graph(self) -> Plan:
        # A zero-task plan is valid: the orchestrator reads it as "nothing to do".
        known: set[str] = set()
        duplicates: set[str] = set()
        for task in self.tasks:
            if task.id in known:
                duplicates.add(task.id)
            known.add(task.id)
        if duplicates:
            raise ValueError(f"plan has duplicate task ids: {', '.join(sorted(duplicates))}")

        for task in self.tasks:
            missing = sorted({dep for dep in task.dependencies if dep not in known})
            if missing:
                raise ValueError(
                    f"task {task.id!r} depends on unknown task ids: {', '.join(missing)}"
                )
        return self


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    # extra="ignore" so unrelated keys in a shared .env do not break startup.
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env", extra="ignore")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-sonnet-4-20250514"
    worker_model: str = "claude-sonnet-4-20250514"
    tester_model: str = "claude-sonnet-4-20250514"
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)  # 0 means run cycles without a limit
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    llm_timeout: float = Field(default=120.0, gt=0)  # per LLM request, seconds
    llm_max_retries: int = Field(default=3, ge=0)  # retries for transient provider errors
    llm_max_tokens: int = Field(default=8192, gt=0)  # max output tokens per request
    test_timeout: float = Field(default=300.0, gt=0)  # whole test-suite timeout, seconds
    test_command: str | None = None  # explicit test command override
    web_host: str = "127.0.0.1"  # loopback only; never 0.0.0.0, it would expose the agent
    web_port: int = Field(default=8000, ge=1, le=65535)

    @model_validator(mode="after")
    def _cross_field_consistency(self) -> Settings:
        # Deliberately no API-key requirement: the module-level `settings = Settings()`
        # below runs at import time, so `import furrow` has to keep working with no
        # credentials configured. Each backend resolves and reports its own key at the
        # point of use instead.
        blank = sorted(
            name
            for name, model_id in (
                ("model", self.model),
                ("planner_model", self.planner_model),
                ("worker_model", self.worker_model),
                ("tester_model", self.tester_model),
            )
            if not model_id.strip()
        )
        if blank:
            raise ValueError(f"model id(s) must not be blank: {', '.join(blank)}")
        if self.test_command is not None and not self.test_command.strip():
            raise ValueError("test_command must not be blank; omit it to auto-detect the suite")
        return self


settings = Settings()


def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return settings
