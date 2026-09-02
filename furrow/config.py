from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

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


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env")

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"

    # API keys (optional; will fall back to env vars at runtime)
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"

    # Development loop settings
    max_parallel_tasks: int = 5
    max_cycles: int = 0  # 0 = unlimited
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    # LLM defaults
    default_max_tokens: int = 8192
    llm_timeout: float = 60.0
    llm_max_retries: int = 3

    # Test runner settings
    test_timeout: int = 120

    # Web server
    web_host: str = "0.0.0.0"
    web_port: int = 8000


settings = Settings()
