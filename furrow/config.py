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


class State(BaseModel):
    goal: str
    cycles: int = 0
    max_cycles: int = 0
    tasks: list[TaskModel] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)


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


settings = Settings()


DEFAULT_STATE_PATH: Path = Path.cwd() / ".furrow" / "state.json"


def state_path(workspace: Path | None = None) -> Path:
    ws = workspace if workspace is not None else Path.cwd()
    path = ws / ".furrow" / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_state(workspace: Path | None = None) -> State:
    path = state_path(workspace)
    try:
        data = path.read_text()
    except FileNotFoundError:
        return State()
    if not data.strip():
        return State()
    try:
        return State.model_validate_json(data)
    except (ValueError, TypeError):
        return State()


def save_state(state: State, workspace: Path | None = None) -> None:
    path = state_path(workspace)
    path.write_text(state.model_dump_json())
