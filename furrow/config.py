from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

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

    # Provider settings
    provider: Provider = Provider.ANTHROPIC
    
    # Model settings
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"
    
    # API Keys
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    
    # Execution settings
    max_parallel_tasks: int = Field(default=5, ge=1, le=20)
    max_cycles: int = Field(default=0, ge=0)  # 0 = unlimited
    max_worker_rounds: int = Field(default=3, ge=1, le=10)
    max_tester_fix_rounds: int = Field(default=3, ge=1, le=10)
    
    # Workspace
    workspace: Path = Field(default_factory=Path.cwd)
    
    # State persistence
    state_file: str = ".furrow_state.json"
    persist_state: bool = True
    
    # Logging
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$")
    
    # Web server
    web_host: str = "0.0.0.0"
    web_port: int = Field(default=8000, ge=1, le=65535)
    
    @field_validator("workspace", mode="before")
    @classmethod
    def _validate_workspace(cls, v: str | Path) -> Path:
        if isinstance(v, str):
            return Path(v).expanduser().resolve()
        return v.expanduser().resolve()
    
    @field_validator("state_file", mode="before")
    @classmethod
    def _validate_state_file(cls, v: str) -> str:
        if not v.endswith(".json"):
            return f"{v}.json"
        return v


settings = Settings()