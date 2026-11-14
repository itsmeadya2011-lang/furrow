from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

from furrow.costs import ModelPricing

DEFAULT_CONFIG_FILE = "furrow.yaml"


def _dotenv_values(
    settings_cls: type[BaseSettings], init_kwargs: dict[str, Any]
) -> dict[str, str]:
    """Read the configured .env file(s) so the config path can be resolved before load."""
    if init_kwargs.get("_env_file", ...) is None:
        return {}
    env_file = settings_cls.model_config.get("env_file")
    if env_file is None:
        return {}
    files = [env_file] if isinstance(env_file, (str, os.PathLike)) else list(env_file)
    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - python-dotenv is a declared dependency
        return {}
    values: dict[str, str] = {}
    for file in files:
        path = Path(file)
        if path.is_file():
            values.update({k: v for k, v in dotenv_values(path).items() if v is not None})
    return values


def _resolve_config_files(
    settings_cls: type[BaseSettings],
    init_kwargs: dict[str, Any],
    dotenv_values: dict[str, str],
) -> list[Path]:
    """Locate the YAML config file relative to the workspace.

    The workspace and config file name may themselves come from init arguments,
    environment variables or the .env file. Only values known before the YAML
    file is read can be used here.
    """

    def lookup(name: str) -> Any | None:
        if name in init_kwargs and init_kwargs[name] is not None:
            return init_kwargs[name]
        value = os.environ.get(f"FURROW_{name.upper()}")
        if value is not None:
            return value
        value = dotenv_values.get(f"FURROW_{name.upper()}")
        if value is not None:
            return value
        return None

    workspace_value = lookup("workspace")
    workspace = Path(workspace_value) if workspace_value is not None else Path.cwd()

    config_value = lookup("config_file")
    config_name = str(config_value) if config_value is not None else DEFAULT_CONFIG_FILE
    config_path = Path(config_name).expanduser()
    if not config_path.is_absolute():
        config_path = workspace / config_path

    candidates = [config_path]
    # When the default name is used, also accept the .yml spelling.
    if config_name == DEFAULT_CONFIG_FILE:
        alternative = config_path.with_suffix(".yml")
        if alternative != config_path:
            candidates.append(alternative)
    return candidates


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: str = "pending"
    result: str | None = None
    error: str | None = None
    attempts: int = 0
    started_at: str | None = None
    finished_at: str | None = None
    files_changed: list[str] = Field(default_factory=list)

    @field_validator("status")
    @classmethod
    def _normalize_status(cls, v: str) -> str:
        if v is None:
            return "pending"
        if not isinstance(v, str):
            return "pending"
        known = {"pending", "running", "completed", "failed", "blocked", "skipped"}
        s = v.strip().lower()
        if s in known:
            return s
        return "pending"


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    __test__ = False
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    skipped: bool = False


class WorkerResult(BaseModel):
    summary: str = ""
    success: bool = True
    files_changed: list[str] = Field(default_factory=list)
    error: str | None = None
    iterations: int = 0


class Usage(BaseModel):
    """Token usage accumulated over one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output) across the counted calls."""
        return self.input_tokens + self.output_tokens

    def add(self, other: Usage) -> Usage:
        """Accumulate *other* into this usage and return self."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls
        self.cost_usd += other.cost_usd
        return self


class RunResult(BaseModel):
    passed: bool = False
    cycles: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_blocked: int = 0
    tests_run: int = 0
    summary: str = ""
    stopped_reason: str = ""
    skipped: bool = False
    usage: Usage = Field(default_factory=Usage)
    files_changed: list[str] = Field(default_factory=list)


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
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    max_worker_iterations: int = Field(default=12, ge=1)
    max_retries: int = Field(default=3, ge=0)
    max_task_retries: int = Field(default=1, ge=0)
    request_timeout: float = Field(default=120.0, gt=0)
    command_timeout: float = Field(default=300.0, gt=0)
    test_timeout: float = Field(default=600.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    max_cost_usd: float = Field(default=0.0, ge=0)
    state_file: Path = Path(".furrow/state.json")
    model_pricing: dict[str, ModelPricing] = Field(default_factory=dict)
    plugins: list[str] = Field(default_factory=list)
    config_file: Path = Path(DEFAULT_CONFIG_FILE)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Load from init args, environment variables, .env, then the YAML file.

        Sources earlier in the tuple take precedence, so environment variables
        (``FURROW_*``) override values from ``furrow.yaml``.
        """
        init_kwargs = getattr(init_settings, "init_kwargs", {}) or {}
        dotenv_values = _dotenv_values(settings_cls, init_kwargs)
        yaml_source = YamlConfigSettingsSource(
            settings_cls,
            yaml_file=_resolve_config_files(settings_cls, init_kwargs, dotenv_values),
        )
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            yaml_source,
            file_secret_settings,
        )

    def state_path(self, workspace: Path | None = None) -> Path:
        """Absolute path of the run-state file: (workspace or self.workspace) / self.state_file"""
        base = Path(workspace) if workspace is not None else Path(self.workspace)
        return base / Path(self.state_file)

    def pricing_overrides(self) -> dict[str, ModelPricing]:
        """User-supplied price overrides that take precedence over the built-in table."""
        return dict(self.model_pricing)


settings = Settings()
