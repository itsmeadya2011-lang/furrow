from __future__ import annotations

from enum import Enum
from pathlib import Path

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
    result: str | None = None


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
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = Field(default=5, ge=1, le=20)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    log_json: bool = False
    state_file: Path = Field(default_factory=lambda: Path.cwd() / ".furrow_state.json")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=100, le=100000)
    test_timeout: int = Field(default=120, ge=10, le=600)
    auto_approve: bool = False

    @field_validator("provider", mode="before")
    @classmethod
    def validate_provider(cls, v: str | Provider) -> Provider:
        if isinstance(v, str):
            try:
                return Provider(v.lower())
            except ValueError:
                raise ValueError(
                    f"Invalid provider: {v}. Must be one of: {[p.value for p in Provider]}"
                )
        return v

    @field_validator("workspace", mode="before")
    @classmethod
    def validate_workspace(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v).resolve()
        return v.resolve()

    @field_validator("state_file", mode="before")
    @classmethod
    def validate_state_file(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v).resolve()
        return v.resolve()


settings = Settings()
