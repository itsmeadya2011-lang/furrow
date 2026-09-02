from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    """LLM provider backends supported by Furrow."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskModel(BaseModel):
    """A single unit of work planned by the PlannerAgent."""

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None


class Plan(BaseModel):
    """A plan produced by the PlannerAgent."""

    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    """The result of running the test suite after a cycle."""

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class FileOperation(BaseModel):
    """A single file operation requested by the WorkerAgent.

    ``operation`` selects the mode:

    - ``"write"``: requires ``content`` (full file contents).
    - ``"edit"``: requires ``old_str`` and ``new_str`` (targeted replacement).
    """

    path: str
    operation: Literal["write", "edit"]
    content: Optional[str] = None
    old_str: Optional[str] = None
    new_str: Optional[str] = None

    @model_validator(mode="after")
    def _validate_operation(self) -> FileOperation:
        if self.operation == "write":
            if self.content is None:
                raise ValueError("FileOperation: 'write' requires 'content'")
        elif self.operation == "edit":
            if self.old_str is None or self.new_str is None:
                raise ValueError("FileOperation: 'edit' requires both 'old_str' and 'new_str'")
        return self


class WorkerResult(BaseModel):
    """Structured result returned by the WorkerAgent."""

    summary: str
    operations: list[FileOperation] = Field(default_factory=list)
    issues: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Global application settings, env-prefixed with ``FURROW_``."""

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
