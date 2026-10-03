from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskModel(BaseModel):
    # LLMs routinely invent extra keys; never let that break construction.
    model_config = ConfigDict(extra="ignore")

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None


class Plan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # pytest tries to collect any imported class named ``Test*``; this model is not
    # a test. Marking it un-collectable silences PytestCollectionWarning without
    # renaming the public model or disabling test-class collection repo-wide.
    __test__ = False

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


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
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()
