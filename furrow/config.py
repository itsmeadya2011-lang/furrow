"""Configuration models for furrow.

Every :class:`Settings` field can be supplied through an environment variable or
through a ``.env`` file located in the current working directory.  By default the
``FURROW_`` prefix is used, e.g. ``FURROW_PROVIDER=openai`` or
``FURROW_MODEL=gpt-4o``.  The two API key fields additionally accept the bare,
provider native spelling documented in the README (``ANTHROPIC_API_KEY`` /
``OPENAI_API_KEY``), so an ``.env`` file written to match the README works.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "ANTHROPIC_DEFAULT_MODELS",
    "MODEL_ROLE_FIELDS",
    "PRISTINE_ANTHROPIC_MODELS",
    "PROVIDER_DEFAULT_MODELS",
    "Plan",
    "Provider",
    "Settings",
    "TaskModel",
    "TaskStatus",
    "TestResult",
    "settings",
]


class Provider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"


class TaskStatus(str, Enum):
    """Lifecycle of a planned task.

    Subclasses :class:`str`, so ``task.status == "completed"`` and
    ``task.status in ("pending", "completed")`` keep working for callers that
    still compare against raw strings.
    """

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"

    def __str__(self) -> str:  # so f"{status}" prints "pending", not "TaskStatus.PENDING"
        return self.value


class TaskModel(BaseModel):
    # validate_assignment keeps ``task.status = "completed"`` working while
    # coercing it to a real TaskStatus member (and rejecting typos).
    model_config = ConfigDict(validate_assignment=True)

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    # Tells pytest that this is a data model, not a test case to collect.
    __test__ = False

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


#: Model fields on :class:`Settings` that carry a model id.
MODEL_ROLE_FIELDS: tuple[str, ...] = ("model", "planner_model", "worker_model", "tester_model")

#: Declared (pristine) Anthropic defaults.  A model field still holding one of
#: these values is considered "not configured by the user" and is therefore
#: swapped for the selected provider's default when the provider is not
#: Anthropic (see :meth:`Settings._apply_provider_defaults`).
ANTHROPIC_DEFAULT_MODELS: dict[str, str] = {
    "model": "claude-sonnet-4-20250514",
    "planner_model": "claude-3-5-haiku-20241022",
    "worker_model": "claude-3-5-sonnet-20241022",
    "tester_model": "claude-3-5-sonnet-20241022",
}

#: Default model id per provider and per role.
PROVIDER_DEFAULT_MODELS: dict[Provider, dict[str, str]] = {
    Provider.ANTHROPIC: dict(ANTHROPIC_DEFAULT_MODELS),
    Provider.OPENAI: {
        "model": "gpt-4o",
        "planner_model": "gpt-4o-mini",
        "worker_model": "gpt-4o",
        "tester_model": "gpt-4o",
    },
    Provider.OLLAMA: {
        "model": "qwen2.5-coder",
        "planner_model": "qwen2.5",
        "worker_model": "qwen2.5-coder",
        "tester_model": "qwen2.5",
    },
}

#: Frozen set of every pristine Anthropic default, used to detect model fields
#: the user never configured explicitly.
PRISTINE_ANTHROPIC_MODELS: frozenset[str] = frozenset(ANTHROPIC_DEFAULT_MODELS.values())


class Settings(BaseSettings):
    """Runtime configuration for furrow.

    Attributes:
        provider: Which LLM backend to talk to.
        model: Default model id for ``LLMClient.complete()`` when no role
            specific model is requested.
        planner_model / worker_model / tester_model: Model ids used by the
            planner, worker and tester agents.
        anthropic_api_key / openai_api_key: Provider credentials.  Accepted from
            the environment or ``.env`` as ``FURROW_ANTHROPIC_API_KEY`` or the
            bare ``ANTHROPIC_API_KEY`` (same for OpenAI).
        ollama_base_url: Base URL of the local Ollama server.
        request_timeout: Per-request timeout in seconds for every provider
            client.  Guards against a stalled socket hanging a whole cycle.
        max_tokens: Output token budget applied on every provider path.
        max_parallel_tasks: Upper bound on concurrently running worker tasks;
            must be >= 1.
        max_cycles: Upper bound on orchestration cycles; must be >= 0.
            **``max_cycles == 0`` means unlimited** (the default).
        workspace: Root directory furrow is allowed to work in.
        log_level: Logging level name.
    """

    model_config = SettingsConfigDict(
        env_prefix="FURROW_",
        env_file=".env",
        env_file_encoding="utf-8",
        populate_by_name=True,
        validate_assignment=True,
        extra="ignore",
    )

    provider: Provider = Provider.ANTHROPIC
    model: str = ANTHROPIC_DEFAULT_MODELS["model"]
    planner_model: str = ANTHROPIC_DEFAULT_MODELS["planner_model"]
    worker_model: str = ANTHROPIC_DEFAULT_MODELS["worker_model"]
    tester_model: str = ANTHROPIC_DEFAULT_MODELS["tester_model"]
    anthropic_api_key: str | None = Field(
        default=None,
        # validation_alias disables env_prefix, so the prefixed spelling has to be
        # listed explicitly; the bare spelling keeps README-style .env files working.
        validation_alias=AliasChoices("FURROW_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("FURROW_OPENAI_API_KEY", "OPENAI_API_KEY"),
    )
    ollama_base_url: str = "http://localhost:11434"
    request_timeout: float = Field(default=120.0, gt=0)
    max_tokens: int = Field(default=4096, ge=1)
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=0, ge=0)  # 0 == unlimited
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> Settings:
        """Swap pristine Anthropic model defaults for the selected provider's defaults.

        Without this, ``FURROW_PROVIDER=openai`` would send ``claude-sonnet-4-...``
        to the OpenAI API and fail upstream with an opaque ``model_not_found``.
        A model the user configured explicitly (anything that is not one of the
        declared Anthropic defaults) is never touched.
        """
        if self.provider is Provider.ANTHROPIC:
            return self
        defaults = PROVIDER_DEFAULT_MODELS.get(self.provider)
        if defaults is None:
            return self
        for role in MODEL_ROLE_FIELDS:
            if getattr(self, role) in PRISTINE_ANTHROPIC_MODELS:
                setattr(self, role, defaults[role])
        return self


settings = Settings()