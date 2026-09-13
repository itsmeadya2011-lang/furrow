from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, Field, SecretStr, field_validator
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


def _default_workspace() -> Path:
    return Path.cwd().resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    anthropic_api_key: Annotated[SecretStr | None, Field(default=None)] = None
    openai_api_key: Annotated[SecretStr | None, Field(default=None)] = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: Annotated[int, Field(ge=1, le=100)] = 5
    max_cycles: Annotated[int, Field(ge=0, le=1000)] = 25
    workspace: Path = Field(default_factory=_default_workspace)
    web_token: Annotated[SecretStr | None, Field(default=None)] = None
    log_level: str = "INFO"
    request_timeout: Annotated[float, Field(gt=0, le=300)] = 30.0
    max_retries: Annotated[int, Field(ge=0, le=10)] = 3
    retry_base_delay: Annotated[float, Field(gt=0, le=60)] = 1.0

    @field_validator("workspace", mode="before")
    @classmethod
    def _resolve_workspace(cls, v: str | Path) -> Path:
        return Path(v).resolve()

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_ollama_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://")
        return v.rstrip("/")

    def get_anthropic_key(self) -> str | None:
        if self.anthropic_api_key is not None:
            return self.anthropic_api_key.get_secret_value()
        return None

    def get_openai_key(self) -> str | None:
        if self.openai_api_key is not None:
            return self.openai_api_key.get_secret_value()
        return None

    def get_web_token(self) -> str | None:
        if self.web_token is not None:
            return self.web_token.get_secret_value()
        return None

    def model_dump(self, *, mode: str = "json", **kwargs: Any) -> dict[str, Any]:
        return super().model_dump(mode=mode, **kwargs)


settings = Settings()
