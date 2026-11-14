"""Custom exceptions for Furrow.

This module defines a hierarchy of exceptions that can be raised during
Furrow operations, allowing for more granular error handling and reporting.
"""

from __future__ import annotations

from typing import Any


class FurrowError(Exception):
    """Base exception for all Furrow errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (details: {self.details})"
        return self.message


class ConfigurationError(FurrowError):
    """Raised when there is a configuration error."""


class ProviderError(FurrowError):
    """Raised when there is an error with an LLM provider."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.provider = provider
        self.model = model
        self.status_code = status_code


class RateLimitError(ProviderError):
    """Raised when the provider returns a rate limit error."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        provider: str | None = None,
        model: str | None = None,
        retry_after: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider, model, 429, details)
        self.retry_after = retry_after


class AuthenticationError(ProviderError):
    """Raised when authentication with the provider fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        provider: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider, None, 401, details)


class ModelNotFoundError(ProviderError):
    """Raised when the requested model is not found."""

    def __init__(
        self,
        message: str = "Model not found",
        provider: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, provider, model, 404, details)


class ValidationError(FurrowError):
    """Raised when input validation fails."""


class SecurityError(FurrowError):
    """Raised when a security policy violation is detected."""

    def __init__(
        self,
        message: str = "Security policy violation",
        command: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.command = command
        self.reason = reason


class PathTraversalError(SecurityError):
    """Raised when a path traversal attempt is detected."""

    def __init__(
        self,
        message: str = "Path traversal attempt detected",
        path: str | None = None,
        workspace: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, path, "path_traversal", details)
        self.path = path
        self.workspace = workspace


class CommandDeniedError(SecurityError):
    """Raised when a command is denied by the security policy."""

    def __init__(
        self,
        message: str = "Command denied by security policy",
        command: str | None = None,
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, command, reason, details)


class TimeoutError(FurrowError):
    """Raised when an operation times out."""

    def __init__(
        self,
        message: str = "Operation timed out",
        operation: str | None = None,
        timeout: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.operation = operation
        self.timeout = timeout


class StateError(FurrowError):
    """Raised when there is an error with state persistence."""

    def __init__(
        self,
        message: str = "State error",
        path: str | None = None,
        operation: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.path = path
        self.operation = operation


class TestRunnerError(FurrowError):
    """Raised when there is an error running tests."""

    def __init__(
        self,
        message: str = "Test runner error",
        command: list[str] | None = None,
        returncode: int | None = None,
        stdout: str | None = None,
        stderr: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.command = command
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class OrchestrationError(FurrowError):
    """Raised when there is an error during orchestration."""

    def __init__(
        self,
        message: str = "Orchestration error",
        cycle: int | None = None,
        task_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details)
        self.cycle = cycle
        self.task_id = task_id


# Exception mapping for HTTP status codes
HTTP_STATUS_MAP: dict[type[FurrowError], int] = {
    ConfigurationError: 500,
    ProviderError: 502,
    RateLimitError: 429,
    AuthenticationError: 401,
    ModelNotFoundError: 404,
    ValidationError: 400,
    SecurityError: 403,
    PathTraversalError: 403,
    CommandDeniedError: 403,
    TimeoutError: 504,
    StateError: 500,
    TestRunnerError: 500,
    OrchestrationError: 500,
}


def get_http_status(error: Exception) -> int:
    """Get the HTTP status code for a Furrow error."""
    for exc_type, status in HTTP_STATUS_MAP.items():
        if isinstance(error, exc_type):
            return status
    return 500