from __future__ import annotations


class FurrowError(Exception):
    """Base exception for Furrow."""


class PlanParseError(FurrowError):
    """Raised when the planner output cannot be parsed."""


class TaskExecutionError(FurrowError):
    """Raised when a task fails to execute."""


class TestError(FurrowError):
    """Raised when tests fail or cannot be run."""