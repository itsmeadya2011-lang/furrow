"""Shared exception taxonomy for Furrow.

Every error raised by Furrow inherits from :class:`FurrowError` so callers can
catch the whole family at once. Errors carry an optional ``details`` mapping that
is rendered into ``str(exc)`` for readable logs without losing the original data.
"""

from __future__ import annotations

from typing import Any

#: Upper bound on how much of a raw model response is echoed into an error message.
_MAX_RAW_ECHO = 500


class FurrowError(Exception):
    """Base class for all Furrow errors."""

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = details

    def __str__(self) -> str:
        if not self.details:
            return self.message
        rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(self.details.items()))
        return f"{self.message} ({rendered})"


class ConfigurationError(FurrowError):
    """Invalid or missing configuration."""


class MissingCredentialsError(ConfigurationError):
    """An API credential required by the selected provider is not configured."""


class LLMError(FurrowError):
    """The LLM provider call failed."""


class LLMResponseError(LLMError):
    """The provider returned a response with no usable text content."""


class StructuredOutputError(FurrowError):
    """Could not parse structured (JSON) output from a model response."""

    def __init__(self, message: str, *, raw: str = "", **details: Any) -> None:
        self.raw = raw
        details.setdefault("raw", raw[:_MAX_RAW_ECHO])
        super().__init__(message, **details)


class ToolError(FurrowError):
    """A tool invocation failed."""

    def __init__(
        self,
        message: str,
        *,
        tool: str = "",
        arguments: dict[str, Any] | None = None,
        **details: Any,
    ) -> None:
        self.tool = tool
        self.arguments: dict[str, Any] = dict(arguments) if arguments else {}
        details.setdefault("tool", tool)
        details.setdefault("arguments", self.arguments)
        super().__init__(message, **details)


class WorkspaceContainmentError(ToolError):
    """A path escaped the configured workspace root."""


class TestExecutionError(FurrowError):
    """The project's test suite could not be executed."""
