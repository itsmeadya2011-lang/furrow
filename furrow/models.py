"""Domain models shared by the planner, worker, tester and orchestrator.

This module is pure schema: it imports nothing from ``furrow`` and has no import-time
side effects, so it can be imported from anywhere (including settings modules) without
pulling in provider SDKs or I/O.

Design note on relaxed defaults
-------------------------------
``Plan.rationale`` and ``TestResult.passed`` are intentionally *optional* (empty string /
``False``) rather than required. Both are produced by an LLM, and a model that omits a
"narrative" field should not invalidate an otherwise usable plan or test report. Missing
``rationale`` degrades to ``""`` and missing ``passed`` degrades to the conservative
``False`` (assume tests did not pass). Both keep the models tolerant of terse output
without weakening validation on the identifying fields.

``TaskModel`` deliberately carries no ``model_config``. In particular
``validate_assignment`` is *not* enabled: the orchestrator assigns plain strings to
``status``/``result`` while a cycle is in flight (including ``None`` from a cancelled
task), and assignment-time validation would reject those mid-run mutations. Because
``TaskStatus`` is a ``str`` enum, assigned strings still compare equal to the enum
members in ``t.status == "completed"`` checks.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

__all__ = ["FileChange", "Plan", "TaskModel", "TaskStatus", "TestResult", "WorkerResult"]


class TaskStatus(str, Enum):
    """Lifecycle state of a single planned task (a str enum, so raw strings compare equal)."""

    #: Queued, not started yet.
    PENDING = "pending"
    #: Handed to a worker.
    RUNNING = "running"
    #: Worker finished without raising.
    COMPLETED = "completed"
    #: Worker or test raised.
    FAILED = "failed"
    #: Intentionally not executed (e.g. superseded or blocked).
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    """One unit of work produced by the planner."""

    #: Stable identifier used to wire up ``dependencies``.
    id: str
    #: Natural-language instruction handed to the worker agent.
    description: str
    #: Repository-relative paths the task is expected to touch.
    files: list[str] = Field(default_factory=list)
    #: IDs of tasks that must complete before this one starts.
    dependencies: list[str] = Field(default_factory=list)
    #: Current lifecycle state; defaults to :attr:`TaskStatus.PENDING`.
    status: TaskStatus = TaskStatus.PENDING
    #: Free-form outcome text filled in by the worker or orchestrator.
    result: str | None = None


class Plan(BaseModel):
    """An ordered batch of tasks produced for a single goal."""

    #: Tasks to execute. May be empty, which the orchestrator reads as "goal complete".
    tasks: list[TaskModel] = Field(default_factory=list)
    #: Model's explanation of the approach. Optional; see the module docstring.
    rationale: str = ""


class TestResult(BaseModel):
    """Outcome of running the project's test suite."""

    #: Whether the suite passed. Defaults to ``False`` when the model omits it.
    passed: bool = False
    #: Human-readable one-line outcome.
    summary: str = ""
    #: Individual failure messages, used to seed the next cycle's goal.
    failures: list[str] = Field(default_factory=list)
    #: True when no runnable test suite was detected, so ``passed`` is meaningless rather
    #: than a real failure. Callers must not treat this as a regression.
    skipped: bool = Field(default=False, description="True = no runnable test suite was detected")


class FileChange(BaseModel):
    """A single file write or deletion requested by the worker."""

    #: Repository-relative path to modify.
    path: str
    #: Full new file contents; ignored when ``action`` is ``"delete"``.
    content: str
    #: Operation to apply: ``"write"`` creates or overwrites, ``"delete"`` removes.
    action: str = Field(default="write", description='"write" | "delete"')


class WorkerResult(BaseModel):
    """What a worker agent reports back after attempting a task."""

    #: One-line statement of what was done.
    summary: str = ""
    #: Concrete edits to apply to the workspace.
    files: list[FileChange] = Field(default_factory=list)
    #: Caveats, follow-ups or blockers the orchestrator should surface.
    notes: str = ""
