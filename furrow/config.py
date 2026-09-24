from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, TomlConfigSettingsSource


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


def _find_config_file() -> Path | None:
    """Find the config file in the current or parent directories."""
    for path in [Path.cwd()] + list(Path.cwd().parents):
        for name in ["furrow.toml", ".furrow.toml", "pyproject.toml"]:
            config_path = path / name
            if config_path.exists():
                return config_path
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FURROW_",
        env_file=".env",
        extra="ignore",
    )

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

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: TomlConfigSettingsSource,
        env_settings: TomlConfigSettingsSource,
        dotenv_settings: TomlConfigSettingsSource,
        file_secret_settings: TomlConfigSettingsSource,
    ) -> tuple[TomlConfigSettingsSource, ...]:
        config_file = _find_config_file()
        if config_file and config_file.suffix == ".toml":
            # If it's pyproject.toml, look for [tool.furrow] section
            if config_file.name == "pyproject.toml":
                try:
                    return (
                        TomlConfigSettingsSource(settings_cls, toml_file=config_file, toml_table_header=("tool", "furrow")),
                        env_settings,
                        dotenv_settings,
                        file_secret_settings,
                    )
                except KeyError:
                    # Section not found, fall back to env/dotenv settings
                    pass
            else:
                return (
                    TomlConfigSettingsSource(settings_cls, toml_file=config_file),
                    env_settings,
                    dotenv_settings,
                    file_secret_settings,
                )
        return (
            env_settings,
            dotenv_settings,
            file_secret_settings,
        )


settings = Settings()