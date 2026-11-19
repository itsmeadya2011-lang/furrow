"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
"""

from __future__ import annotations

import logging
import sys
import uuid
from contextvars import ContextVar
from typing import Any


# Context variables for adding contextual information to logs
run_id_var: ContextVar[str | None] = ContextVar("run_id", default=None)
cycle_var: ContextVar[int | None] = ContextVar("cycle", default=None)
task_id_var: ContextVar[str | None] = ContextVar("task_id", default=None)


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


def _add_context_vars(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add context variables to log entries."""
    run_id = run_id_var.get()
    cycle = cycle_var.get()
    task_id = task_id_var.get()
    
    if run_id:
        event_dict["run_id"] = run_id
    if cycle is not None:
        event_dict["cycle"] = cycle
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
            _add_context_vars,
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


def set_run_context(run_id: str | None = None, cycle: int | None = None, task_id: str | None = None) -> None:
    """Set context variables for structured logging."""
    if run_id is not None:
        run_id_var.set(run_id)
    if cycle is not None:
        cycle_var.set(cycle)
    if task_id is not None:
        task_id_var.set(task_id)


def clear_run_context() -> None:
    """Clear all run context variables."""
    run_id_var.set(None)
    cycle_var.set(None)
    task_id_var.set(None)


def generate_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())[:8]
