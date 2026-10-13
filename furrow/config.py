from __future__ import annotations

from enum import Enum
from pathlib import Path

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
    result: str | None = None
    cycle: int = 0
    attempts: int = 0
    error: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class TokenUsage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost: float = 0.0


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
    enable_cost_tracking: bool = True
    cost_per_million_input: dict[str, float] = Field(default_factory=lambda: {
        "claude-sonnet-4-20250514": 3.00,
        "claude-3-5-haiku-20241022": 0.25,
        "claude-3-5-sonnet-20241022": 3.00,
        "gpt-4o": 5.00,
        "gpt-4o-mini": 0.15,
    })
    cost_per_million_output: dict[str, float] = Field(default_factory=lambda: {
        "claude-sonnet-4-20250514": 15.00,
        "claude-3-5-haiku-20241022": 1.25,
        "claude-3-5-sonnet-20241022": 15.00,
        "gpt-4o": 15.00,
        "gpt-4o-mini": 0.60,
    })


settings = Settings()
