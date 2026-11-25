"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


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
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(sort_keys=True),
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


def bind_run_context(logger: Any, run_id: str, goal: str, cycle: int = 0) -> Any:
    """Bind run context to a logger for structured logging."""
    if _STRUCTLOG_AVAILABLE:
        return logger.bind(run_id=run_id, goal=goal, cycle=cycle)
    return logger


def bind_task_context(logger: Any, task_id: str, task_description: str) -> Any:
    """Bind task context to a logger for structured logging."""
    if _STRUCTLOG_AVAILABLE:
        return logger.bind(task_id=task_id, task_description=task_description)
    return logger


def log_event(logger: Any, event: str, **kwargs: Any) -> None:
    """Log a structured event with context."""
    if _STRUCTLOG_AVAILABLE:
        logger.info(event, **kwargs)
    else:
        # Fallback for stdlib logging
        ctx = " ".join(f"{k}={v}" for k, v in kwargs.items())
        logger.info(f"{event} {ctx}")
