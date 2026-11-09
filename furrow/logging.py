"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
"""

from __future__ import annotations

import contextvars
import logging
import sys
import uuid
from typing import Any

# Context variable for correlation ID tracking across async boundaries
correlation_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)

# Context variable for run ID tracking
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "run_id", default=None
)

# Context variable for cycle ID tracking
cycle_id_var: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "cycle_id", default=None
)

# Context variable for task ID tracking
task_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "task_id", default=None
)


def generate_correlation_id() -> str:
    """Generate a new correlation ID."""
    return uuid.uuid4().hex[:12]


def generate_run_id() -> str:
    """Generate a new run ID."""
    return f"run_{uuid.uuid4().hex[:8]}"


def get_correlation_id() -> str | None:
    """Get the current correlation ID."""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set the correlation ID, generating one if not provided."""
    if correlation_id is None:
        correlation_id = generate_correlation_id()
    correlation_id_var.set(correlation_id)
    return correlation_id


def get_run_id() -> str | None:
    """Get the current run ID."""
    return run_id_var.get()


def set_run_id(run_id: str | None = None) -> str:
    """Set the run ID, generating one if not provided."""
    if run_id is None:
        run_id = generate_run_id()
    run_id_var.set(run_id)
    return run_id


def get_cycle_id() -> int | None:
    """Get the current cycle ID."""
    return cycle_id_var.get()


def set_cycle_id(cycle_id: int | None) -> None:
    """Set the cycle ID."""
    cycle_id_var.set(cycle_id)


def get_task_id() -> str | None:
    """Get the current task ID."""
    return task_id_var.get()


def set_task_id(task_id: str | None) -> None:
    """Set the task ID."""
    task_id_var.set(task_id)


def _configure_stdlib_fallback(level: str = "INFO") -> None:
    """Configure standard library logging as a fallback."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


def _add_context_processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add correlation ID and other context to log entries."""
    corr_id = correlation_id_var.get()
    if corr_id:
        event_dict["correlation_id"] = corr_id
    run_id = run_id_var.get()
    if run_id:
        event_dict["run_id"] = run_id
    cycle_id = cycle_id_var.get()
    if cycle_id is not None:
        event_dict["cycle_id"] = cycle_id
    task_id = task_id_var.get()
    if task_id:
        event_dict["task_id"] = task_id
    return event_dict


try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore[assignment]
    _STRUCTLOG_AVAILABLE = False


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
            _add_context_processor,
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
