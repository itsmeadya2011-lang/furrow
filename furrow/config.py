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


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


class TaskModel(BaseModel):
    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None

    def start(self) -> "TaskModel":
        return transition_start(self)

    def complete(self, result: Optional[str] = None) -> "TaskModel":
        return transition_complete(self, result)

    def fail(self, error: Optional[str] = None) -> "TaskModel":
        return transition_fail(self, error)

    def block(self, reason: Optional[str] = None) -> "TaskModel":
        return transition_block(self, reason)


def transition_start(task: TaskModel) -> TaskModel:
    """Move a task into the running state, clearing any prior error."""
    task.status = TaskStatus.RUNNING
    task.error = None
    return task


def transition_complete(task: TaskModel, result: Optional[str] = None) -> TaskModel:
    """Mark a task as successfully completed with an optional result."""
    task.status = TaskStatus.COMPLETED
    task.result = result
    task.error = None
    return task


def transition_fail(task: TaskModel, error: Optional[str] = None) -> TaskModel:
    """Mark a task as failed, recording the error message."""
    task.status = TaskStatus.FAILED
    task.error = error
    return task


def transition_block(task: TaskModel, reason: Optional[str] = None) -> TaskModel:
    """Mark a task as blocked, recording the reason."""
    task.status = TaskStatus.BLOCKED
    task.error = reason
    return task


class Plan(BaseModel):
    tasks: list[TaskModel]
    rationale: str

    def validate(self) -> None:
        validate_task_graph(self.tasks)


class TestResult(BaseModel):
    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    command: Optional[list[str]] = None
    exit_code: Optional[int] = None


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
    max_parallel_tasks: int = Field(default=5, ge=1)
    max_cycles: int = Field(default=10, ge=1)
    test_command: list[str] = Field(
        default_factory=lambda: ["python", "-m", "pytest", "-q"]
    )
    test_timeout_seconds: int = Field(default=120, gt=0)
    workspace: Path = Field(default_factory=Path.cwd)
    log_level: str = "INFO"


def validate_task_graph(tasks: list[TaskModel]) -> None:
    """Validate a list of tasks forming a dependency graph.

    Raises ValueError for:
      - empty or duplicate task IDs
      - dependencies referencing unknown task IDs
      - self-dependencies
      - cycles in the dependency graph
    """
    ids = [t.id for t in tasks]

    # Empty IDs
    empty = [tid for tid in ids if tid is None or str(tid).strip() == ""]
    if empty:
        raise ValueError(f"Task graph contains empty IDs: {empty}")

    # Duplicate IDs
    seen: set[str] = set()
    duplicates: list[str] = []
    for tid in ids:
        if tid in seen:
            duplicates.append(tid)
        seen.add(tid)
    if duplicates:
        raise ValueError(f"Task graph contains duplicate IDs: {duplicates}")

    # Unknown dependencies and self-dependencies
    for task in tasks:
        for dep in task.dependencies:
            if dep not in seen:
                raise ValueError(
                    f"Task '{task.id}' depends on unknown task '{dep}'"
                )
            if dep == task.id:
                raise ValueError(f"Task '{task.id}' depends on itself")

    # Cycle detection via DFS
    adjacency: dict[str, list[str]] = {tid: [] for tid in ids}
    for task in tasks:
        adjacency[task.id] = list(task.dependencies)

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {tid: WHITE for tid in ids}

    def dfs(node: str) -> bool:
        color[node] = GRAY
        for neighbor in adjacency[node]:
            if color[neighbor] == GRAY:
                return False
            if color[neighbor] == WHITE and not dfs(neighbor):
                return False
        color[node] = BLACK
        return True

    for tid in ids:
        if color[tid] == WHITE:
            if not dfs(tid):
                raise ValueError("Task graph contains a dependency cycle")


settings = Settings()