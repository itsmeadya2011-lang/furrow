"""Configuration for furrow.

Settings are resolved lazily through `get_settings()` rather than at import
time. Importing this module used to read `.env` from whatever directory the
process happened to be in, which made every consumer (tests, CLI, web server)
inherit ambient configuration as a side effect of an `import` statement.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from furrow.models import Plan, TaskModel, TaskStatus, TestResult, WorkerResult


class Provider(str, Enum):
    """Which LLM backend is in use.

    Modelled as a `str` enum so the value survives logging and JSON round-trips
    unchanged, while comparisons in `furrow.llm` stay exhaustive.
    """

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class Settings(BaseSettings):
    """Everything furrow needs to run, sourced from the environment and `.env`.

    API keys are `SecretStr` rather than `str` on purpose: a plain string field
    is printed verbatim by pydantic's `__repr__` and by any structured logger
    that dumps the settings object, which is how keys ended up in tracebacks and
    logs. Consumers must therefore call `.get_secret_value()` at the point of
    use and never log the result.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="FURROW_",
        env_prefix_target="variable",
        extra="ignore",
        validate_assignment=True,
        # Allows `Settings(anthropic_api_key=...)` by field name even though the
        # two key fields declare an explicit validation_alias below.
        populate_by_name=True,
    )

    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-5"
    # Per-role overrides; None means "inherit `model`", which keeps `--model` on
    # the CLI able to seed every role at once instead of being silently ignored
    # by the three pinned defaults this replaces.
    planner_model: str | None = None
    worker_model: str | None = None
    tester_model: str | None = None

    # An explicit `validation_alias` takes precedence over `env_prefix`, so the
    # FURROW_-prefixed spelling has to be listed by hand. It is listed first so
    # that it wins when both names are present. Without this, a user's existing
    # bare ANTHROPIC_API_KEY / OPENAI_API_KEY - exactly what the README tells
    # them to export - would never bind and the run would fail on a missing key.
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    openai_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
        validation_alias=AliasChoices("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    # Ceiling on worker agents running at once. Lowering it trades throughput
    # for a smaller blast radius when the LLM provider rate-limits.
    max_parallel_tasks: int = Field(default=5, ge=1)
    # How many dependency waves one cycle may execute before the cycle ends and
    # a fresh plan is taken. This caps work per cycle, not per run: a plan with
    # ten waves will stop after `max_parallel_waves` of them and be replanned.
    max_parallel_waves: int = Field(default=3, ge=1)
    # 0 means UNLIMITED cycles, not "no cycles". This is the single most
    # misreadable setting in the project, so the sentinel is explicit: any other
    # negative value is rejected rather than silently treated as unlimited.
    max_cycles: int = Field(default=0, ge=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"
    request_timeout: float = Field(default=120.0, gt=0)
    test_command_timeout: float = Field(default=300.0, gt=0)
    max_tokens: int = Field(default=8192, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    @field_validator("provider", mode="before")
    @classmethod
    def _normalise_provider(cls, value: object) -> object:
        """Accept any capitalisation of the provider name.

        `.env` files are written by hand, so `FURROW_PROVIDER=Anthropic` is a
        normal thing to type and pydantic's enum matching is case-sensitive.
        Unrecognised values are passed through untouched so the user gets
        pydantic's standard "valid values are ..." message instead of a
        KeyError from this validator.
        """
        if isinstance(value, str):
            try:
                return Provider[value.strip().upper()]
            except KeyError:
                return value
        return value

    def model_for(self, role: str) -> str:
        """Resolve the model for a role, falling back to the global model.

        Roles are "planner", "worker", "tester". An explicit per-role override
        wins; otherwise the single `model` setting is used. This collapses three
        previously-duplicated ternary expressions into one place and makes
        `--model` on the CLI able to seed all roles at once.
        """
        explicit = getattr(self, f"{role}_model", None)
        return explicit or self.model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide Settings, building them on first use.

    Cached so every component shares one object, but built lazily so importing
    `furrow.config` never touches the filesystem or the environment. Tests and
    the CLI call `reset_settings_cache()` after changing configuration to pick
    the change up.
    """
    return Settings()


def reset_settings_cache() -> None:
    """Drop the cached Settings so the next `get_settings()` re-resolves them.

    Needed by anything that mutates the environment or a `.env` file at
    runtime, most notably the CLI applying flags and tests isolating themselves.
    """
    get_settings.cache_clear()


# Re-exported so existing importers of `furrow.config.Plan` and friends keep
# working now that the models have their own module.
__all__ = [
    "Plan",
    "Provider",
    "Settings",
    "TaskModel",
    "TaskStatus",
    "TestResult",
    "WorkerResult",
    "get_settings",
    "reset_settings_cache",
]
