from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: Optional[str] = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


SessionStatus = Literal["running", "paused", "completed"]


class SessionState(BaseModel):
    """Persistent state for a Furrow orchestration session.

    Stores enough information to resume an in-progress orchestration
    after a restart. The current plan is stored as a serialized
    dictionary so that the model schema can evolve without breaking
    older session files.
    """

    session_id: str
    goal: str
    current_goal: str
    cycles: int = 0
    current_plan: Optional[dict[str, Any]] = None
    status: SessionStatus = "running"
    workspace: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        """Update the ``updated_at`` timestamp to the current time."""
        self.updated_at = datetime.now(timezone.utc)

    def to_json(self) -> str:
        """Serialize the session state to a JSON string."""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> "SessionState":
        """Deserialize a ``SessionState`` from a JSON string."""
        return cls.model_validate_json(data)


class Settings(BaseSettings):
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
