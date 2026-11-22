"""Error handling utilities for Furrow.

This module provides structured error types and handling utilities for the
Furrow system.
"""

from __future__ import annotations

from enum import Enum
from typing import Any


class FurrowError(Exception):
    """Base exception for Furrow errors."""

    def __init__(self, message: str, code: str = "FURROW_ERROR", details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}


class ConfigurationError(FurrowError):
    """Configuration-related errors."""
    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message, "CONFIGURATION_ERROR", details)


class ProviderError(FurrowError):
    """LLM provider errors."""
    def __init__(self, message: str, provider: str, details: dict[str, Any] | None = None):
        super().__init__(message, "PROVIDER_ERROR", {"provider": provider, **(details or {})})


class ToolError(FurrowError):
    """Tool execution errors."""
    def __init__(self, message: str, tool: str, details: dict[str, Any] | None = None):
        super().__init__(message, "TOOL_ERROR", {"tool": tool, **(details or {})})


class SecurityError(FurrowError):
    """Security policy violations."""
    def __init__(self, message: str, command: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message, "SECURITY_ERROR", {"command": command, **(details or {})})


class StateError(FurrowError):
    """State persistence errors."""
    def __init__(self, message: str, path: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message, "STATE_ERROR", {"path": path, **(details or {})})


class TimeoutError(FurrowError):
    """Timeout errors."""
    def __init__(self, message: str, operation: str, timeout: float, details: dict[str, Any] | None = None):
        super().__init__(message, "TIMEOUT_ERROR", {"operation": operation, "timeout": timeout, **(details or {})})


class ValidationError(FurrowError):
    """Validation errors."""
    def __init__(self, message: str, field: str | None = None, details: dict[str, Any] | None = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, **(details or {})})


class ErrorCode(str, Enum):
    """Standard error codes."""
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    TOOL_ERROR = "TOOL_ERROR"
    SECURITY_ERROR = "SECURITY_ERROR"
    STATE_ERROR = "STATE_ERROR"
    TIMEOUT_ERROR = "TIMEOUT_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


def format_error(error: Exception) -> dict[str, Any]:
    """Format an exception for logging/serialization."""
    if isinstance(error, FurrowError):
        return {
            "code": error.code,
            "message": error.message,
            "details": error.details,
        }
    return {
        "code": ErrorCode.UNKNOWN_ERROR,
        "message": str(error),
        "details": {"type": type(error).__name__},
    }


def handle_error(error: Exception, logger=None) -> dict[str, Any]:
    """Handle an error and return formatted error info."""
    formatted = format_error(error)
    if logger:
        logger.error(
            "Furrow error: %s",
            formatted["message"],
            code=formatted["code"],
            details=formatted["details"],
        )
    return formatted