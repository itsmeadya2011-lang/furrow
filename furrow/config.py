"""Configuration and shared data models for Furrow.

Every environment variable uses the ``FURROW_`` prefix (for example
``FURROW_PROVIDER=openai``, ``FURROW_MODEL=gpt-4o``, ``FURROW_MAX_CYCLES=5``)
and values may also come from a ``.env`` file in the current working directory.
Env var names are case-insensitive.

Notable semantics:

* ``max_cycles`` -- ``0`` (the default) means **unlimited** cycles; any positive
  value bounds the orchestrator loop to that many cycles.
* ``fail_fast_cycles`` -- how many *consecutive* cycles may make no progress before
  the loop halts instead of retrying forever. A cycle counts when it raised an unexpected
  exception or finished with a failing test run; a cycle whose tests pass -- or that planned
  no tasks -- resets the streak. Non-positive values are clamped to 1, so ``0`` cannot
  switch the bound off.
* ``test_timeout`` -- seconds allowed for a single test command run.
* ``max_tool_turns`` -- maximum tool-call rounds a worker gets per task.
* ``test_command`` -- optional explicit test command; when unset the tester picks
  a runner by probing well-known commands.

The module-level :data:`settings` object is built at import time, but it must
never break ``import furrow``: a malformed environment or ``.env`` file (say
``FURROW_PROVIDER=foo``) is downgraded to a logged warning and the defaults are
used instead of raising a pydantic ``ValidationError`` at import.
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

ENV_PREFIX = "FURROW_"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Lifecycle of a single planned task.

    Subclasses ``str``, so both ``task.status == "completed"`` and
    ``task.status is TaskStatus.COMPLETED`` hold. Callers may freely assign plain
    strings (``task.status = "failed"``) without breaking comparisons.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    status_message: str | None = None
    result: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix=ENV_PREFIX, env_file=".env")

    # -- models ---------------------------------------------------------
    provider: Provider = Provider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    planner_model: str = "claude-3-5-haiku-20241022"
    worker_model: str = "claude-3-5-sonnet-20241022"
    tester_model: str = "claude-3-5-sonnet-20241022"

    # -- credentials ----------------------------------------------------
    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"

    # -- orchestration limits -------------------------------------------
    max_parallel_tasks: int = 5
    max_cycles: int = 0  # 0 == unlimited
    fail_fast_cycles: int = 3  # consecutive non-progressing cycles before halting
    max_tool_turns: int = 10  # tool-call rounds allowed per worker task

    # -- testing ---------------------------------------------------------
    test_timeout: int = 120
    test_command: str | None = None

    # -- workspace / logging ---------------------------------------------
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


def configure_logging(level: str | None = None) -> None:
    """Configure the root logger once with a readable format.

    Idempotent: if the root logger already has handlers (a host application, a
    test runner, or a previous call) this is a no-op, so embedding Furrow never
    clobbers the caller's logging setup.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=(level or settings.log_level).upper(), format=LOG_FORMAT)


def _invalid_field_names(exc: ValidationError) -> list[str]:
    """Extract the field names pydantic rejected from a ``ValidationError``."""
    names: list[str] = []
    for error in exc.errors():
        loc = error.get("loc") or ()
        if len(loc) == 1 and isinstance(loc[0], str):
            names.append(loc[0])
    return names


def _load_settings() -> Settings:
    """Build the global settings without ever raising during ``import furrow``.

    ``_env_file`` is a real ``pydantic_settings.BaseSettings`` initialiser kwarg
    (it overrides the ``env_file`` from :data:`Settings.model_config`), but
    pydantic-settings ships no stub for it, so mypy reports it as an unexpected
    argument. The ignore below is scoped to that one gap; enabling the pydantic
    mypy plugin in ``pyproject.toml`` is the alternative when that file is in scope.
    """
    try:
        return Settings()
    except ValidationError as exc:
        logger.warning("Invalid Furrow configuration (%s); retrying without .env file.", exc)

    try:
        return Settings(_env_file=None)  # type: ignore[call-arg]  # pydantic-settings init kwarg
    except ValidationError as exc:
        logger.warning("Invalid %s* environment variable(s) (%s); using defaults.", ENV_PREFIX, exc)
        fields = _invalid_field_names(exc)

    # Last resort: the bad values came from the environment itself, so temporarily
    # hide the offending variables while building defaults, then restore them.
    hidden: dict[str, str] = {}
    try:
        for name in fields:
            key = f"{ENV_PREFIX}{name.upper()}"
            if key in os.environ:
                hidden[key] = os.environ.pop(key)
        return Settings(_env_file=None)  # type: ignore[call-arg]  # pydantic-settings init kwarg
    finally:
        os.environ.update(hidden)


settings = _load_settings()
