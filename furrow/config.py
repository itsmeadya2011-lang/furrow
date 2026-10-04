from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None

    def short(self) -> str:
        return f"[{self.id}] {self.description}"


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    """The verdict of one verification cycle.

    ``verified`` is ``False`` when no test runner could be detected in the
    workspace, in which case ``passed`` carries no real signal at all: an
    unverifiable workspace must not be read as a passing one.
    """

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    verified: bool = True

    # Not a pytest class: the "Test" prefix would otherwise make pytest try to
    # collect this model as a test case. pytest honours this attribute.
    __test__ = False


class Settings(BaseSettings):
    """Furrow configuration.

    A module-level ``settings`` singleton is constructed at import time for
    convenience, but tests should construct their own ``Settings`` instances
    to avoid leaking environment state between test cases.
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
    max_parallel_tasks: int = Field(default=5, ge=1, le=32)
    max_cycles: int = Field(default=0, ge=0)  # 0 means unlimited cycles
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    def model_for(self, role: str) -> str:
        """Resolve the model name for a given role.

        Falls back to the default ``model`` for unknown roles.
        """
        if role == "planner":
            return self.planner_model
        if role == "worker":
            return self.worker_model
        if role == "tester":
            return self.tester_model
        return self.model

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, value: str) -> str:
        return value.upper()


settings = Settings()
