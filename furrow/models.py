"""Domain models for furrow.

These live in their own module because the orchestrator, the agents and the
persistence layer all need to agree on the shape of a plan, a task and a test
result, while `furrow.config` only cares about how those values are configured.
Keeping them here means a test can import a plain pydantic model without
triggering any settings resolution.
"""

from __future__ import annotations

from enum import Enum
from typing import ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskStatus(str, Enum):
    """Lifecycle state of a single unit of planned work.

    Modelled as a `str` enum rather than a bare `str` because the value is
    round-tripped through the orchestrator, serialised into JSON by the planner
    LLM and compared by the wave scheduler. A plain `str` field silently accepts
    "Pending" and "PENDING", which then look like a task that never runs.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class TaskModel(BaseModel):
    """One unit of work handed from the planner to a worker.

    `dependencies` holds ids of other tasks in the same plan; the scheduler uses
    them to decide which tasks may start together, so a typo in the list is
    indistinguishable from "no dependencies" unless it is rejected at parse time.
    """

    id: str
    description: str
    files: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str | None = None
    attempts: int = 0

    @field_validator("files", "dependencies")
    @classmethod
    def _reject_blank_entries(cls, value: list[str]) -> list[str]:
        """Strip surrounding whitespace and reject empty entries.

        Paths and task ids arrive from an LLM that regularly emits trailing
        spaces and the occasional `""`. A blank path would be resolved against
        the workspace root and a blank dependency would never be satisfied,
        producing a plan that looks complete but stalls, so both are refused
        here rather than debugged in the orchestrator.
        """
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("entries must be non-empty, whitespace-only values are rejected")
        return cleaned


class Plan(BaseModel):
    """The planner agent's output: a set of tasks plus why it chose them.

    Every field defaults so an empty plan is constructible: an LLM answering
    "nothing to do" must produce a valid model rather than a validation error,
    because an exception here would be indistinguishable from a broken planner.
    """

    tasks: list[TaskModel] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def _reject_duplicate_ids(self) -> Plan:
        """Refuse plans that name the same task id twice.

        Duplicate ids break the dependency graph: the scheduler would mark one
        of them complete twice and the results would be attributed to whichever
        task finished last. Catching it at parse time turns a silently wrong run
        into an explicit planner error.
        """
        seen: set[str] = set()
        for task in self.tasks:
            if task.id in seen:
                raise ValueError(f"duplicate task id: {task.id!r}")
            seen.add(task.id)
        return self


class TestResult(BaseModel):
    """Outcome of running the project's test suite.

    `skipped` exists because a skipped run is NOT a pass. When no test runner
    could be discovered the tester used to report the literal string
    "No test runner found." as the summary; the LLM read that as a test report
    and cheerfully returned `passed: true`, so projects with no tests were
    reported as green. Callers must therefore check `skipped` before trusting
    `passed`, and must never infer either from `summary` text.
    """

    passed: bool
    summary: str
    failures: list[str] = Field(default_factory=list)
    # Deliberately independent of `passed`: `skipped=True` with `passed=False`
    # means "unknown", never "green". See the class docstring.
    skipped: bool = False
    command: str | None = None

    # pytest collects any class whose name starts with "Test" and warns when it
    # cannot build a collector from it. This attribute is pytest's documented
    # opt-out, so importing TestResult into a test module stays warning-free.
    __test__: ClassVar[bool] = False


class WorkerResult(BaseModel):
    """What a worker actually did, as opposed to what it said it would do.

    `written_files` and `issue` exist so the orchestrator can verify work
    independently of the worker's prose: an LLM that claims success without
    changing a single byte is a failure mode that prose alone cannot detect.
    """

    summary: str = ""
    written_files: list[str] = Field(default_factory=list)
    issue: str | None = None
