"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.

Provides correlation IDs, run IDs, and structured context for traceability.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from contextlib import contextmanager
from typing import Any

# Context variables for correlation tracking
_correlation_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("correlation_id", default=None)
_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)
_cycle_id: contextvars.ContextVar[int | None] = contextvars.ContextVar("cycle_id", default=None)
_task_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)


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


def _add_correlation_context(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add correlation IDs to every log event."""
    corr_id = _correlation_id.get()
    if corr_id:
        event_dict["correlation_id"] = corr_id
    run_id = _run_id.get()
    if run_id:
        event_dict["run_id"] = run_id
    cycle = _cycle_id.get()
    if cycle is not None:
        event_dict["cycle"] = cycle
    task_id = _task_id.get()
    if task_id:
        event_dict["task_id"] = task_id
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to wrap stdlib logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    if not _STRUCTLOG_AVAILABLE:
        # Fall back to stdlib logging if structlog is not available
        _configure_stdlib_fallback(level)
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            _add_correlation_context,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
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


def new_correlation_id() -> str:
    """Generate a new correlation ID."""
    return uuid.uuid4().hex[:16]


def new_run_id() -> str:
    """Generate a new run ID."""
    return uuid.uuid4().hex[:12]


@contextmanager
def correlation_context(correlation_id: str | None = None, run_id: str | None = None,
                       cycle: int | None = None, task_id: str | None = None):
    """Context manager to set correlation context for structured logging.

    Args:
        correlation_id: Optional correlation ID (generated if not provided)
        run_id: Optional run ID (generated if not provided)
        cycle: Optional cycle number
        task_id: Optional task ID

    Example:
        with correlation_context(run_id="abc123", cycle=1, task_id="task-1"):
            logger.info("Starting task")
    """
    corr_token = _correlation_id.set(correlation_id or new_correlation_id())
    run_token = _run_id.set(run_id or new_run_id())
    cycle_token = _cycle_id.set(cycle)
    task_token = _task_id.set(task_id)
    try:
        yield
    finally:
        _correlation_id.reset(corr_token)
        _run_id.reset(run_token)
        _cycle_id.reset(cycle_token)
        _task_id.reset(task_token)


def get_correlation_id() -> str | None:
    """Get the current correlation ID."""
    return _correlation_id.get()


def get_run_id() -> str | None:
    """Get the current run ID."""
    return _run_id.get()


def get_cycle_id() -> int | None:
    """Get the current cycle number."""
    return _cycle_id.get()


def get_task_id() -> str | None:
    """Get the current task ID."""
    return _task_id.get()
