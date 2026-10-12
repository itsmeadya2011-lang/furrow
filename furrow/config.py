from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator
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
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env", extra="ignore")

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

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v_upper

    @field_validator("max_parallel_tasks")
    @classmethod
    def validate_max_parallel_tasks(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_parallel_tasks must be at least 1")
        if v > 50:
            raise ValueError("max_parallel_tasks cannot exceed 50")
        return v

    @field_validator("ollama_base_url")
    @classmethod
    def validate_ollama_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://")
        return v

    @model_validator(mode="after")
    def validate_api_keys(self) -> "Settings":
        if self.provider == Provider.ANTHROPIC and not self.anthropic_api_key:
            import os
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise ValueError("ANTHROPIC_API_KEY is required when provider is anthropic")
        elif self.provider == Provider.OPENAI and not self.openai_api_key:
            import os
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OPENAI_API_KEY is required when provider is openai")
        return self


class _LazySettings:
    """Lazy proxy around Settings().

    Deferring instantiation until first access avoids requiring API keys (or
    any other env var) at import time, which lets the package be imported in
    environments where credentials are not yet configured (e.g. test collection).
    """

    def __init__(self) -> None:
        self._instance: Settings | None = None

    def _get(self) -> Settings:
        if self._instance is None:
            self._instance = Settings()
        return self._instance

    def __getattr__(self, item: str):
        return getattr(self._get(), item)

    def __setattr__(self, key: str, value) -> None:
        if key == "_instance":
            object.__setattr__(self, key, value)
        else:
            setattr(self._get(), key, value)

    def __repr__(self) -> str:
        if self._instance is None:
            return "<LazySettings (uninitialized)>"
        return repr(self._instance)


settings = _LazySettings()
