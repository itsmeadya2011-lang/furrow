"""Custom exceptions for Furrow.

This module defines a hierarchy of exceptions used throughout the Furrow
codebase to provide more specific error handling and better error messages.
"""

from __future__ import annotations


class FurrowError(Exception):
    """Base exception for all Furrow errors."""

    def __init__(self, message: str, *, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


class ConfigurationError(FurrowError):
    """Raised when there is a configuration error."""

    pass


class PlanningError(FurrowError):
    """Raised when the planner fails to create a valid plan."""

    pass


class ExecutionError(FurrowError):
    """Raised when task execution fails."""

    pass


class TestingError(FurrowError):
    """Raised when the test runner fails."""

    pass


class StateError(FurrowError):
    """Raised when there is an error reading/writing state."""

    pass


class SecurityError(FurrowError):
    """Raised when a security policy violation is detected."""

    pass


class LLMError(FurrowError):
    """Raised when an LLM provider error occurs."""

    pass


class TimeoutError(FurrowError):
    """Raised when an operation times out."""

    pass


class ValidationError(FurrowError):
    """Raised when input validation fails."""

    pass


def wrap_error(exc: Exception, context: str) -> FurrowError:
    """Wrap an exception with additional context.

    Args:
        exc: The original exception.
        context: A description of what was happening when the error occurred.

    Returns:
        A FurrowError with the context included in the message.
    """
    if isinstance(exc, FurrowError):
        # Already a FurrowError, just add context to the message
        return FurrowError(f"{context}: {exc}", cause=exc.cause)
    return FurrowError(f"{context}: {type(exc).__name__}: {exc}", cause=exc)