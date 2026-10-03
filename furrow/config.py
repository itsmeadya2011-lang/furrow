from __future__ import annotations

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
    # pytest collects classes named Test*; without this every module importing the
    # model emits "cannot collect test class 'TestResult'". Pydantic v2 leaves the
    # underscore attribute alone (it is not a field and not a private attr), so
    # TestResult(passed=True, summary="s") is unchanged.
    __test__ = False

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)


class Settings(BaseSettings):
    """Runtime configuration.

    Every field has a default so ``Settings()`` works with no environment set up;
    values can be overridden per instance, via ``FURROW_*`` env vars, or in a
    ``.env`` file (see ``env_prefix``/``env_file``).
    """

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

    # Generation knobs. 4096 tokens covers a full plan/task response; temperature
    # stays low (not 0.0) because providers may reject or degenerate at exactly 0.
    max_tokens: int = 4096
    temperature: float = 0.2
    # Networking: per-request timeout, retries for transient provider failures
    # (rate limit/5xx/timeout/connection reset) and a cap on the backoff sleep.
    request_timeout: float = 120.0
    max_retries: int = 2
    max_retry_wait: float = 20.0
    # Wall-clock budget for a single test-runner subprocess.
    test_timeout: int = 120

    # Applying worker edits to the workspace. apply_changes=False is advisory mode:
    # the worker still reports what it would have written but touches nothing.
    apply_changes: bool = True
    backup_dir: Path = Path(".furrow/backups")
    # Per-task limits on a single worker's writes. Each accepted write is a whole-file
    # replacement, so the caps bound how much of the repo one reply can rewrite.
    max_files_per_task: int = 10
    max_write_bytes: int = 200_000
    max_total_write_bytes: int = 1_000_000
    stagnation_threshold: float = 0.8
    max_stagnant_cycles: int = 3


settings = Settings()
