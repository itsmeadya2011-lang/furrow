from __future__ import annotations

from enum import Enum
from pathlib import Path

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
    result: str | None = None


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str


class TestResult(BaseModel):
    # Not a pytest test class: pytest collects any imported class named Test* and
    # warns about this one because it has an __init__. Keep the marker so the
    # warning stays gone.
    __test__ = False

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


#: Every model field a ``--model`` override (CLI or web console) has to rewrite.
#: ``PlannerAgent``, ``WorkerAgent`` and ``TesterAgent`` each pass their own
#: ``planner_model`` / ``worker_model`` / ``tester_model`` to
#: ``LLMClient.complete``, so the generic ``model`` fallback is never read by any
#: of them: overriding only that one field changes nothing at runtime, which is a
#: dead override rather than a partial one.  Both entry points therefore share
#: this list instead of each naming the fields separately.
MODEL_FIELDS = ("model", "planner_model", "worker_model", "tester_model")


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
    max_parallel_tasks: int = 5
    max_cycles: int = 0
    test_timeout: int = 300
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


settings = Settings()


def settings_for_model(model: str | None) -> Settings | None:
    """Copy of the global settings with every agent model set to ``model``.

    Returns ``None`` when there is no override, so the caller keeps using the
    module-level singleton and nothing changes for the common case.

    The global ``settings`` object is never mutated: it is shared process-wide
    (and by ``furrow web``'s concurrent runs), so a copy is handed to one run
    instead.  Mutating the singleton would let one browser tab re-point every
    other concurrent run -- and the CLI -- at its model, permanently, with no way
    to undo it.

    An explicit override deliberately wins over ``FURROW_PLANNER_MODEL`` and
    friends.  The decision rule is "more specific, more recent instruction wins":
    those variables are ambient configuration chosen when the environment was set
    up, while a flag (or the web console's model box) was typed by the user just
    now, naming this run.  A user who wants per-role models from the environment
    simply omits the override.
    """
    if not model:
        return None
    return settings.model_copy(update=dict.fromkeys(MODEL_FIELDS, model))
