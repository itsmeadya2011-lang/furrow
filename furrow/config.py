from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv.load_dotenv()


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

    @field_validator("workspace", mode="before")
    @classmethod
    def validate_workspace_exists(cls, v: Path | str) -> Path:
        path = Path(v)
        if not path.exists() or not path.is_dir():
            raise ValueError(f"workspace must be an existing directory, got: {path}")
        return path

    @field_validator("anthropic_api_key", "openai_api_key")
    @classmethod
    def validate_api_key_for_cloud_providers(cls, v: Optional[str], info) -> Optional[str]:
        provider = info.data.get("provider")
        if provider in (Provider.ANTHROPIC, Provider.OPENAI):
            if not v:
                raise ValueError(
                    f"{provider.value} provider requires an API key to be set"
                )
        return v


settings = Settings()
