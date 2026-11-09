"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
"""

from __future__ import annotations

import contextvars
import logging
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any

# Context variables for run/task context
run_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "run_context", default=None
)


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


def _add_run_context(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """Add run context (cycle, task_id, etc.) to log entries."""
    ctx = run_context.get()
    if ctx:
        event_dict.update(ctx)
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
            _add_run_context,
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


def set_run_context(**kwargs: Any) -> contextvars.Token:
    """Set run context variables (cycle, task_id, etc.) for structured logging.
    
    Returns a token that can be used to reset the context.
    """
    ctx = run_context.get() or {}
    ctx = ctx.copy()
    ctx.update(kwargs)
    return run_context.set(ctx)


def reset_run_context(token: contextvars.Token) -> None:
    """Reset run context to previous state."""
    run_context.reset(token)
