from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import structlog
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
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    furrow_config_file: str = "furrow.json"

    # --- Project-level config (furrow.json) ---

    @classmethod
    def load_project_config(cls, workspace: Path | None = None) -> dict[str, Any]:
        """Load project-level settings from ``furrow.json``.

        Returns a dict of settings overrides (e.g. ``"max_cycles": 10``).
        Returns an empty dict if the file is missing or invalid.
        """
        ws = workspace or Path.cwd()
        config_path = ws / cls.model_fields["furrow_config_file"].default
        if not config_path.exists():
            return {}
        try:
            data = json.loads(config_path.read_text())
            return data.get("settings", {})
        except (json.JSONDecodeError, OSError) as e:
            logging.getLogger(__name__).warning(
                "Failed to load furrow.json: %s", e
            )
            return {}

    @classmethod
    def from_workspace(cls, workspace: Path | None = None) -> "Settings":
        """Create Settings, merging furrow.json project overrides on top
        of environment variables and defaults."""
        ws = workspace or Path.cwd()
        env_settings = cls(workspace=ws)
        overrides = cls.load_project_config(ws)
        # Apply furrow.json overrides that don't conflict with explicit env vars
        for key, value in overrides.items():
            if key in cls.model_fields and not getattr(env_settings, key, None):
                setattr(env_settings, key, value)
        return env_settings


# Module-level singleton
settings = Settings()


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with console-friendly output.

    Uses a simple processor chain that is robust across structlog versions.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


# Configure logging at import time
configure_logging(settings.log_level)
