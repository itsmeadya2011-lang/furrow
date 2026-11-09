"""Custom exceptions for Furrow."""

from __future__ import annotations


class FurrowError(Exception):
    """Base exception for all Furrow errors."""

    pass


class LLMError(FurrowError):
    """Raised when an LLM provider returns an error or unexpected response."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.retryable = retryable

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.provider:
            parts.append(f"provider={self.provider}")
        if self.model:
            parts.append(f"model={self.model}")
        if self.status_code:
            parts.append(f"status={self.status_code}")
        if self.retryable:
            parts.append("retryable")
        return " ".join(parts)


class LLMProviderError(LLMError):
    """Raised when the LLM provider returns an HTTP error response."""

    def __init__(
        self,
        message: str,
        provider: str,
        model: str | None = None,
        status_code: int | None = None,
        response_body: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            retryable=retryable,
        )
        self.response_body = response_body


class LLMTimeoutError(LLMError):
    """Raised when an LLM request times out."""

    def __init__(self, message: str, provider: str | None = None, model: str | None = None) -> None:
        super().__init__(message, provider=provider, model=model, retryable=True)


class LLMRateLimitError(LLMError):
    """Raised when the LLM provider returns a rate limit error (429)."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(
            message, provider=provider, model=model, status_code=429, retryable=True
        )
        self.retry_after = retry_after


class LLMQuotaExceededError(LLMError):
    """Raised when the LLM provider returns a quota exceeded error (402, 429)."""

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(
            message,
            provider=provider,
            model=model,
            status_code=status_code,
            retryable=False,
        )


class LLMInvalidResponseError(LLMError):
    """Raised when the LLM provider returns an invalid or malformed response."""

    def __init__(self, message: str, provider: str | None = None, model: str | None = None) -> None:
        super().__init__(message, provider=provider, model=model, retryable=False)


class ConfigurationError(FurrowError):
    """Raised when there is a configuration error."""

    pass


class WorkspaceError(FurrowError):
    """Raised when there is a workspace-related error."""

    pass


class StateError(FurrowError):
    """Raised when there is a state persistence error."""

    pass


class TaskError(FurrowError):
    """Raised when a task execution fails."""

    def __init__(
        self, message: str, task_id: str | None = None, attempt: int | None = None
    ) -> None:
        super().__init__(message)
        self.task_id = task_id
        self.attempt = attempt


class ToolError(FurrowError):
    """Raised when a tool execution fails."""

    def __init__(
        self, message: str, tool_name: str | None = None, tool_args: dict | None = None
    ) -> None:
        super().__init__(message)
        self.tool_name = tool_name
        self.tool_args = tool_args


__all__ = [
    "ConfigurationError",
    "FurrowError",
    "LLMError",
    "LLMInvalidResponseError",
    "LLMProviderError",
    "LLMQuotaExceededError",
    "LLMRateLimitError",
    "LLMTimeoutError",
    "StateError",
    "TaskError",
    "ToolError",
    "WorkspaceError",
]