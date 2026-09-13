from __future__ import annotations

import json
from datetime import datetime
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


class Session(BaseModel):
    session_id: str = ""
    goal: str = ""
    plan: Optional[Plan] = None
    cycles: int = 0
    created_at: str = ""
    updated_at: str = ""

    def update_timestamp(self) -> None:
        now = datetime.utcnow().isoformat()
        self.updated_at = now
        if not self.created_at:
            self.created_at = now


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
    session_file: Path = Field(default=".furrow_session.json")

    def save_session(self, session: Session) -> None:
        session.update_timestamp()
        path = self.workspace / self.session_file
        path.write_text(session.model_dump_json(indent=2))

    def load_session(self) -> Optional[Session]:
        path = self.workspace / self.session_file
        if not path.exists():
            return None
        try:
            return Session.model_validate_json(path.read_text())
        except (json.JSONDecodeError, ValueError):
            return None


settings = Settings()
