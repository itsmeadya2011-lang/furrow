from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
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
    result: Any = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env", extra="ignore")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = Field(default=5, ge=1, le=20)
    max_cycles: int = Field(default=0, ge=0)  # 0 = unlimited
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_json: bool = False

    @field_validator("workspace", mode="before")
    @classmethod
    def _resolve_workspace(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v).resolve()
        return v.resolve()

    def validate_api_keys(self) -> None:
        """Validate that required API keys are present for the selected provider."""
        if self.provider == Provider.ANTHROPIC and not (self.anthropic_api_key or __import__("os").getenv("ANTHROPIC_API_KEY")):
            raise ValueError("ANTHROPIC_API_KEY is required for Anthropic provider")
        if self.provider == Provider.OPENAI and not (self.openai_api_key or __import__("os").getenv("OPENAI_API_KEY")):
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")
        if self.provider == Provider.OLLAMA:
            # Ollama doesn't require API key, just base URL
            pass


settings = Settings()
