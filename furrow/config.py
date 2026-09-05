from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator
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

    @model_validator(mode="after")
    def _validate_provider_keys(self) -> "Settings":
        # Defer strict validation to LLMClient.validate() to avoid import-time crashes
        # when API keys are not set in the environment.
        return self

    @model_validator(mode="after")
    def _validate_max_parallel_tasks(self) -> "Settings":
        if not 1 <= self.max_parallel_tasks <= 10:
            raise ValueError("max_parallel_tasks must be between 1 and 10")
        return self

    def detect_project_type(self) -> str:
        ws = Path(self.workspace)
        if (ws / "pyproject.toml").exists() or (ws / "setup.py").exists() or (ws / "requirements.txt").exists():
            return "python"
        if (ws / "package.json").exists():
            return "node"
        if (ws / "Cargo.toml").exists():
            return "rust"
        if (ws / "go.mod").exists():
            return "go"
        return "unknown"

    def get_test_command(self) -> list[str]:
        project_type = self.detect_project_type()
        if project_type == "python":
            return ["python", "-m", "pytest", "-q"]
        if project_type == "node":
            return ["npm", "test", "--", "--silent"]
        if project_type == "rust":
            return ["cargo", "test", "-q"]
        if project_type == "go":
            return ["go", "test", "./..."]
        return ["pytest", "-q"]


settings = Settings()
