from __future__ import annotations

import functools
from enum import Enum
from pathlib import Path

from pydantic import AliasChoices, BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Fallback instructions used when a caller does not supply a system prompt.
DEFAULT_SYSTEM_PROMPT = "You are a helpful coding assistant."


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


#: Model names used when the provider is Anthropic. These double as the sentinels the
#: provider validator below looks for, so they must stay in sync with the field defaults.
ANTHROPIC_MODEL_DEFAULTS: dict[str, str] = {
    "model": "claude-sonnet-4-20250514",
    "planner_model": "claude-3-5-haiku-20241022",
    "worker_model": "claude-3-5-sonnet-20241022",
    "tester_model": "claude-3-5-sonnet-20241022",
}

OPENAI_MODEL_DEFAULTS: dict[str, str] = {
    "model": "gpt-4o",
    "planner_model": "gpt-4o-mini",
    "worker_model": "gpt-4o",
    "tester_model": "gpt-4o-mini",
}

OLLAMA_MODEL_DEFAULTS: dict[str, str] = {
    "model": "qwen2.5-coder:7b",
    "planner_model": "llama3.2:3b",
    "worker_model": "qwen2.5-coder:7b",
    "tester_model": "llama3.2:3b",
}

PROVIDER_MODEL_DEFAULTS: dict[Provider, dict[str, str]] = {
    Provider.ANTHROPIC: ANTHROPIC_MODEL_DEFAULTS,
    Provider.OPENAI: OPENAI_MODEL_DEFAULTS,
    Provider.OLLAMA: OLLAMA_MODEL_DEFAULTS,
}


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
    # Tell pytest this is not a test class, otherwise it emits a PytestCollectionWarning.
    __test__ = False

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    # extra="ignore" is load-bearing: pydantic-settings reads dotenv entries regardless of
    # env_prefix, so with the "forbid" default one unrelated key in a developer's .env
    # (e.g. HOST=...) raised a ValidationError at import time and broke `import furrow`.
    model_config = SettingsConfigDict(env_prefix="FURROW_", env_file=".env", extra="ignore")

    provider: Provider = Provider.ANTHROPIC
    model: str = ANTHROPIC_MODEL_DEFAULTS["model"]
    planner_model: str = ANTHROPIC_MODEL_DEFAULTS["planner_model"]
    worker_model: str = ANTHROPIC_MODEL_DEFAULTS["worker_model"]
    tester_model: str = ANTHROPIC_MODEL_DEFAULTS["tester_model"]
    # Accept both the FURROW_-prefixed spelling and the bare name documented in README.md;
    # an explicit validation_alias disables env_prefix, so both have to be listed.
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        validation_alias=AliasChoices("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_seconds: float = 120.0
    # Bounds each individual test-runner subprocess, not the suite as a whole.
    test_timeout_seconds: float = 120.0
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _apply_provider_model_defaults(self) -> Settings:
        """Point the model names at the configured provider.

        Switching ``provider`` alone would otherwise send ``claude-*`` names to OpenAI or
        Ollama and 404 on the first call. Fields the caller supplied explicitly (init kwarg,
        env var or .env entry) are always left untouched.
        """
        defaults = PROVIDER_MODEL_DEFAULTS[self.provider]
        for name, provider_default in defaults.items():
            if name in self.model_fields_set:
                continue
            if getattr(self, name) == ANTHROPIC_MODEL_DEFAULTS[name]:
                setattr(self, name, provider_default)
        return self


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, validated once and reused."""
    return Settings()


#: Kept as a module-level singleton for existing importers (``furrow.cli.main`` et al).
settings = get_settings()