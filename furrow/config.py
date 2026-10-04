"""Provider selection, application settings, and the cached settings accessor.

This module owns *configuration* only. The pydantic domain schemas (plan, task, test
result, file change) live in :mod:`furrow.models`; they are re-exported at the bottom of
this file purely as a backwards-compatibility shim.
"""

from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Compatibility shim: `agents/planner.py`, `agents/worker.py`, `agents/tester.py`,
# `core/orchestrator.py` and `tests/test_core.py` all still do
# `from furrow.config import TaskModel, Plan, TestResult`. Keep those imports working.
# New code must import the domain models from `furrow.models` instead.
from furrow.models import (  # noqa: F401  (re-exported for backwards compatibility)
    FileChange,
    Plan,
    TaskModel,
    TaskStatus,
    TestResult,
    WorkerResult,
)

__all__ = [
    "FileChange",
    "Plan",
    "Provider",
    "Settings",
    "TaskModel",
    "TaskStatus",
    "TestResult",
    "WorkerResult",
    "get_settings",
    "settings",
]


class Provider(str, Enum):
    """LLM backend to route requests through.

    Derives from :class:`str` so it can be compared to and serialised as its wire value.
    """

    #: Anthropic Claude API.
    ANTHROPIC = "anthropic"
    #: OpenAI chat-completions API.
    OPENAI = "openai"
    #: Local Ollama daemon; no API key required.
    OLLAMA = "ollama"

    @property
    def env_var(self) -> str | None:
        """Environment variable holding this provider's credential, if it needs one."""
        return {
            Provider.ANTHROPIC: "ANTHROPIC_API_KEY",
            Provider.OPENAI: "OPENAI_API_KEY",
            Provider.OLLAMA: None,
        }[self]


# Because `env_prefix="FURROW_"` is global, a field that declares its own
# `validation_alias` is no longer auto-prefixed. `AliasChoices` lists both the prefixed
# and the bare variable explicitly, so FURROW_OPENAI_API_KEY *and* the README-documented
# OPENAI_API_KEY both work. `exclude=True` keeps the secret out of `model_dump()`, which
# is what structured logs serialise.
AnthropicKey = Annotated[
    str | None,
    Field(
        None,
        validation_alias=AliasChoices("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
        exclude=True,
        description="Anthropic API key; read from FURROW_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY",
    ),
]
OpenAIKey = Annotated[
    str | None,
    Field(
        None,
        validation_alias=AliasChoices("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY"),
        exclude=True,
        description="OpenAI API key; read from FURROW_OPENAI_API_KEY or OPENAI_API_KEY",
    ),
]


class Settings(BaseSettings):
    """Application configuration, populated from the environment and ``.env``.

    Fields:
        provider: LLM backend in use.
        model: Fallback model id when a role-specific model is not set.
        planner_model: Model used by the planner agent.
        worker_model: Model used by the worker agent.
        tester_model: Model used by the tester agent.
        anthropic_api_key: Anthropic credential; excluded from ``model_dump()``.
        openai_api_key: OpenAI credential; excluded from ``model_dump()``.
        ollama_base_url: Base URL of the local Ollama daemon.
        ollama_model: Model id passed to Ollama when that provider is selected.
        workspace: Project root override; ``None`` means "the process CWD, resolved at
            call time". Use :meth:`workspace_path` rather than reading this directly.
        max_cycles: Stop after this many planner cycles; ``0`` means unlimited.
        max_parallel_tasks: Upper bound on tasks executed concurrently in one cycle.
        max_tokens: Completion token budget requested from the provider.
        test_command: Explicit test command override, e.g. ``"pytest -q"``; ``None``
            means auto-detect from the project layout.
        test_timeout: Per-run test timeout in seconds.
        log_level: Root log level name.
        log_json: Emit structlog JSON instead of human-readable console output.

    No validation here raises for a missing API key. ``Settings()`` is constructed at
    import time in several places (and by tests that never talk to a provider), so a
    raising ``model_validator`` would break ``import furrow`` for anyone without a key.
    Call :meth:`ensure_api_key` at the point of first use instead.
    """

    model_config = SettingsConfigDict(
        env_prefix="FURROW_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    provider: Provider = Field(default=Provider.ANTHROPIC, description="LLM backend to use")
    model: str = Field(
        default="claude-sonnet-4-20250514", description="Fallback model id for any role"
    )
    planner_model: str = Field(
        default="claude-sonnet-4-20250514", description="Model used by the planner agent"
    )
    worker_model: str = Field(
        default="claude-sonnet-4-20250514", description="Model used by the worker agent"
    )
    tester_model: str = Field(
        default="claude-sonnet-4-20250514", description="Model used by the tester agent"
    )
    anthropic_api_key: AnthropicKey = None
    openai_api_key: OpenAIKey = None
    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL of the local Ollama daemon"
    )
    ollama_model: str = Field(default="llama3.1", description="Model id requested from Ollama")
    workspace: Path | None = Field(
        default=None,
        description="Project root override; None resolves the process CWD at call time",
    )
    max_cycles: int = Field(default=0, description="Maximum planner cycles; 0 means unlimited")
    max_parallel_tasks: int = Field(
        default=4, description="Maximum tasks executed concurrently per cycle"
    )
    max_tokens: int = Field(
        default=8192, description="Completion token budget requested from the provider"
    )
    test_command: str | None = Field(
        default=None,
        description='Explicit test command override, e.g. "pytest -q"; None auto-detects',
    )
    test_timeout: int = Field(default=120, description="Per-run test timeout in seconds")
    log_level: str = Field(default="INFO", description="Root log level name")
    log_json: bool = Field(
        default=False, description="Emit structlog JSON instead of console output"
    )

    # NOTE: pydantic's `Field(exclude=True)` only hides a field from `model_dump()` /
    # `model_dump_json()` — it does NOT affect `__repr__` / `__str__`, which still render
    # the raw value via `BaseModel.__repr_args__`. That let a live API key leak into any
    # log line, traceback, exception message, rich panel or pytest failure output that
    # happened to repr a Settings instance. structlog serialises via model_dump(), so the
    # `exclude=True` comment above was misleadingly "safe for structlog" but unsafe for
    # repr. Override `__repr_args__` (the narrow hook `BaseModel.__repr__` consumes) so
    # the two credential fields are masked in repr output while every other field keeps
    # pydantic's normal rendering. We mask rather than omit so the repr still shows *that*
    # a key is present, which is useful diagnostically without exposing the value.
    _MASKED_SECRET_FIELDS = ("anthropic_api_key", "openai_api_key")

    def __repr_args__(self) -> list[tuple[str | None, object]]:
        args = super().__repr_args__()
        return [
            (name, "***" if name in self._MASKED_SECRET_FIELDS else value)
            for name, value in args
        ]

    def workspace_path(self) -> Path:
        """Absolute project root the agents operate on.

        Defaults to the process CWD, resolved at call time rather than import time, so
        that changing directory after import (or constructing ``Settings`` at module
        scope) still yields the right root.
        """
        return (self.workspace or Path.cwd()).expanduser().resolve()

    def api_key(self) -> str | None:
        """Credential for the selected provider, or ``None`` for Ollama/unset."""
        if self.provider is Provider.ANTHROPIC:
            return self.anthropic_api_key
        if self.provider is Provider.OPENAI:
            return self.openai_api_key
        return None

    def missing_api_key_error(self) -> str:
        """Actionable message describing how to supply the selected provider's key."""
        name = self.provider.env_var
        if name is None:
            return f"Provider {self.provider.value} does not require an API key."
        return (
            f"No API key configured for provider {self.provider.value!r}. "
            f"Set {name} or FURROW_{name} in the environment or a .env file."
        )

    def ensure_api_key(self) -> str:
        """Return the configured API key, raising ``ValueError`` if it is missing.

        Returns an empty string for providers that need no credential (Ollama) so the
        result is always safe to hand to a client constructor.
        """
        key = self.api_key()
        if key:
            return key
        if self.provider.env_var is None:
            return ""
        raise ValueError(self.missing_api_key_error())

    def models(self) -> dict[str, str]:
        """Role-to-model-id mapping handed to the planner, worker and tester agents."""
        return {
            "planner": self.planner_model,
            "worker": self.worker_model,
            "tester": self.tester_model,
        }

    def resolved_model(self, override: str | None = None) -> str:
        """Return ``override`` when given, otherwise the generic :attr:`model`."""
        return override or self.model


@lru_cache(maxsize=1)
def _build_settings() -> Settings:
    """Construct the process-wide settings singleton. Cached; see :func:`get_settings`."""
    return Settings()


def get_settings(*, refresh: bool = False) -> Settings:
    """Return the cached process-wide :class:`Settings`.

    ``refresh=True`` discards the cached instance and rebuilds it from the current
    environment, which tests and CLI ``--model`` style overrides rely on.
    """
    if refresh:
        _build_settings.cache_clear()
    return _build_settings()


# Deprecated: kept for `cli/main.py` and any external caller importing it directly.
# Prefer `get_settings()` so a refresh is actually observable.
settings = get_settings()
