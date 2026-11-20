"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.

Supports:
- JSON output for production (structured, parseable)
- Human-readable output for development
- Correlation IDs for request tracing
- Structured event logging with consistent fields
"""

from __future__ import annotations

import logging
import os
import sys
import uuid
from contextvars import ContextVar
from typing import Any

# Context variable for correlation ID (e.g., run ID, request ID)
_correlation_id: ContextVar[str | None] = ContextVar("_correlation_id", default=None)


def get_correlation_id() -> str | None:
    """Get the current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None) -> None:
    """Set the correlation ID in context."""
    _correlation_id.set(correlation_id)


def new_correlation_id() -> str:
    """Generate and set a new correlation ID."""
    cid = str(uuid.uuid4())[:8]
    set_correlation_id(cid)
    return cid


def _configure_stdlib_fallback(level: str = "INFO") -> None:
    """Configure standard library logging as a fallback."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore[assignment]
    _STRUCTLOG_AVAILABLE = False


def _add_correlation_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add correlation ID to log event if present."""
    cid = get_correlation_id()
    if cid:
        event_dict["correlation_id"] = cid
    return event_dict


def _add_logger_name(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add logger name to event dict."""
    event_dict["logger"] = logger.__class__.__name__ if hasattr(logger, "__class__") else "unknown"
    return event_dict


def configure_logging(level: str = "INFO", json_output: bool | None = None) -> None:
    """Configure structlog to wrap stdlib logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Force JSON output. If None, auto-detect from FURROW_LOG_JSON env var.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Auto-detect JSON output from env if not explicitly set
    if json_output is None:
        json_output = os.getenv("FURROW_LOG_JSON", "").lower() in ("1", "true", "yes")

    # Configure stdlib logging
    if json_output:
        # JSON output: just pass through the JSON from structlog
        logging.basicConfig(
            format="%(message)s",
            level=log_level,
            stream=sys.stdout,
        )
    else:
        # Human-readable output
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(name)s %(message)s",
            level=log_level,
            stream=sys.stdout,
            datefmt="%H:%M:%S",
        )

    if not _STRUCTLOG_AVAILABLE:
        _configure_stdlib_fallback(level)
        return

    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        _add_correlation_id,
        _add_logger_name,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        # Production: JSON output
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: human-readable with colors if available
        try:
            renderer = structlog.dev.ConsoleRenderer(colors=True)
        except Exception:
            renderer = structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "logger", "event", "correlation_id"])

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a structlog logger, falling back to stdlib if needed."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)


def bind_context(**kwargs: Any) -> Any:
    """Bind context variables to the current logger context.
    
    Returns a context manager that clears the bindings on exit.
    """
    if _STRUCTLOG_AVAILABLE:
        return structlog.contextvars.bound_contextvars(**kwargs)
    # Fallback: no-op context manager
    from contextlib import contextmanager
    
    @contextmanager
    def noop():
        yield
    return noop()


def log_event(logger: Any, event: str, **kwargs: Any) -> None:
    """Log a structured event with consistent fields.
    
    This provides a unified way to log events across the application,
    ensuring consistent field names and structure.
    """
    logger.info(event, **kwargs)


def log_error(logger: Any, event: str, error: Exception | str, **kwargs: Any) -> None:
    """Log an error with structured fields."""
    if isinstance(error, Exception):
        logger.error(event, error_type=type(error).__name__, error_message=str(error), **kwargs)
    else:
        logger.error(event, error_message=str(error), **kwargs)
