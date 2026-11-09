"""Structured logging with correlation IDs for Furrow.

This module provides utilities for adding correlation IDs to log entries,
making it easier to trace requests across multiple components and services.
"""

from __future__ import annotations

import contextvars
import logging
import uuid
from collections.abc import MutableMapping
from contextlib import contextmanager
from typing import Any

import structlog

# Context variable for storing the correlation ID
_correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Context variable for storing additional structured context
_structured_context_var: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "structured_context", default=None
)


def get_correlation_id() -> str | None:
    """Get the current correlation ID, if any."""
    return _correlation_id_var.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set the correlation ID for the current context.

    Args:
        correlation_id: Optional correlation ID to use. If not provided, a new
            UUID4 will be generated.

    Returns:
        The correlation ID that was set.
    """
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    _correlation_id_var.set(correlation_id)
    return correlation_id


def clear_correlation_id() -> None:
    """Clear the correlation ID for the current context."""
    _correlation_id_var.set(None)


@contextmanager
def correlation_context(correlation_id: str | None = None):
    """Context manager for setting a correlation ID temporarily.

    Args:
        correlation_id: Optional correlation ID to use. If not provided, a new
            UUID4 will be generated.

    Example:
        with correlation_context("my-request-id"):
            logger.info("Processing request")
    """
    old_id = _correlation_id_var.get()
    new_id = set_correlation_id(correlation_id)
    try:
        yield new_id
    finally:
        if old_id is not None:
            _correlation_id_var.set(old_id)
        else:
            clear_correlation_id()


def get_structured_context() -> dict[str, Any]:
    """Get the current structured context, if any."""
    ctx = _structured_context_var.get()
    return ctx if ctx is not None else {}


def set_structured_context(context: dict[str, Any] | None) -> dict[str, Any]:
    """Set the structured context for the current context.

    Args:
        context: Dictionary of structured context data. If None, clears the context.

    Returns:
        The context that was set (empty dict if None was provided).
    """
    if context is None:
        _structured_context_var.set({})
        return {}
    _structured_context_var.set(context)
    return context


def merge_structured_context(**kwargs: Any) -> dict[str, Any]:
    """Merge additional key-value pairs into the structured context.

    Args:
        **kwargs: Key-value pairs to add to the context.

    Returns:
        The updated context.
    """
    current = get_structured_context()
    current.update(kwargs)
    return set_structured_context(current)


class CorrelationIdProcessor:
    """Structlog processor that adds correlation ID to log entries."""

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        correlation_id = get_correlation_id()
        if correlation_id is not None:
            event_dict["correlation_id"] = correlation_id
        return event_dict


class StructuredContextProcessor:
    """Structlog processor that adds structured context to log entries."""

    def __call__(
        self, logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
    ) -> MutableMapping[str, Any]:
        context = get_structured_context()
        if context:
            event_dict["context"] = context
        return event_dict


def add_correlation_id_processor() -> None:
    """Add the correlation ID processor to structlog's processor chain."""
    structlog.configure(
        processors=[
            *structlog.get_config()["processors"],
            CorrelationIdProcessor(),
            StructuredContextProcessor(),
        ]
    )


def configure_logging_with_correlation(level: str = "INFO") -> None:
    """Configure structlog with correlation ID support.

    This replaces the default configure_logging function to add correlation
    ID and structured context support.
    """
    import logging
    import sys

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    try:
        import structlog

        structlog.configure(
            processors=[
                structlog.contextvars.merge_contextvars,
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                CorrelationIdProcessor(),
                StructuredContextProcessor(),
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
    except ImportError:
        # Fall back to stdlib logging if structlog is not available
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            level=log_level,
            stream=sys.stdout,
        )