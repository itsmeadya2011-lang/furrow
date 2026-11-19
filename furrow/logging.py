"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
Provides structured logging with tracing capabilities for better observability.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from contextlib import contextmanager
from typing import Any

# Context variables for structured logging
_trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)
_span_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("span_id", default=None)
_cycle_var: contextvars.ContextVar[int | None] = contextvars.ContextVar("cycle", default=None)
_task_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("task_id", default=None)


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


def _add_trace_context(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add trace context to log entries."""
    trace_id = _trace_id_var.get()
    span_id = _span_id_var.get()
    cycle = _cycle_var.get()
    task_id = _task_id_var.get()
    
    if trace_id:
        event_dict["trace_id"] = trace_id
    if span_id:
        event_dict["span_id"] = span_id
    if cycle is not None:
        event_dict["cycle"] = cycle
    if task_id:
        event_dict["task_id"] = task_id
    return event_dict


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to wrap stdlib logging with tracing support."""
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
            _add_trace_context,
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


def set_trace_id(trace_id: str | None = None) -> str:
    """Set the trace ID for the current context. Returns the trace ID."""
    if trace_id is None:
        trace_id = str(uuid.uuid4())[:8]
    _trace_id_var.set(trace_id)
    return trace_id


def get_trace_id() -> str | None:
    """Get the current trace ID."""
    return _trace_id_var.get()


def set_span_id(span_id: str | None = None) -> str:
    """Set the span ID for the current context. Returns the span ID."""
    if span_id is None:
        span_id = str(uuid.uuid4())[:8]
    _span_id_var.set(span_id)
    return span_id


def get_span_id() -> str | None:
    """Get the current span ID."""
    return _span_id_var.get()


def set_cycle(cycle: int | None) -> None:
    """Set the current cycle number."""
    _cycle_var.set(cycle)


def get_cycle() -> int | None:
    """Get the current cycle number."""
    return _cycle_var.get()


def set_task_id(task_id: str | None) -> None:
    """Set the current task ID."""
    _task_id_var.set(task_id)


def get_task_id() -> str | None:
    """Get the current task ID."""
    return _task_id_var.get()


@contextmanager
def trace_context(
    trace_id: str | None = None,
    span_id: str | None = None,
    cycle: int | None = None,
    task_id: str | None = None,
):
    """Context manager for setting trace context."""
    old_trace = _trace_id_var.get()
    old_span = _span_id_var.get()
    old_cycle = _cycle_var.get()
    old_task = _task_id_var.get()
    
    try:
        if trace_id is not None:
            _trace_id_var.set(trace_id)
        if span_id is not None:
            _span_id_var.set(span_id)
        if cycle is not None:
            _cycle_var.set(cycle)
        if task_id is not None:
            _task_id_var.set(task_id)
        yield
    finally:
        # Restore previous values
        if old_trace is not None:
            _trace_id_var.set(old_trace)
        else:
            _trace_id_var.set(None)
        if old_span is not None:
            _span_id_var.set(old_span)
        else:
            _span_id_var.set(None)
        if old_cycle is not None:
            _cycle_var.set(old_cycle)
        else:
            _cycle_var.set(None)
        if old_task is not None:
            _task_id_var.set(old_task)
        else:
            _task_id_var.set(None)


@contextmanager
def new_span(name: str, **kwargs: Any):
    """Context manager for creating a new span with logging."""
    span_id = set_span_id()
    logger = get_logger(name)
    logger.info(f"span_start: {name}", **kwargs)
    try:
        yield span_id
    except Exception as e:
        logger.error(f"span_error: {name}", error=str(e), exc_info=True)
        raise
    finally:
        logger.info(f"span_end: {name}")
