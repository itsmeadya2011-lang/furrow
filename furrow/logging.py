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


def configure_logging(
    level: str = "INFO",
    json_output: bool = True,
    include_timestamp: bool = True,
    include_level: bool = True,
    include_logger_name: bool = True,
) -> None:
    """Configure structlog to wrap stdlib logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: Whether to output logs as JSON
        include_timestamp: Whether to include timestamp in logs
        include_level: Whether to include log level in logs
        include_logger_name: Whether to include logger name in logs
    """
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

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
    ]

    if include_level:
        processors.append(structlog.processors.add_log_level)

    if include_timestamp:
        processors.append(structlog.processors.TimeStamper(fmt="iso", utc=True))

    processors.extend([
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ])

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "logger", "event"]))

    structlog.configure(
        processors=processors,
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


def set_log_context(**kwargs: Any) -> None:
    """Set context variables for structured logging.

    These will be included in all subsequent log messages.
    """
    if _STRUCTLOG_AVAILABLE:
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(**kwargs)


def clear_log_context() -> None:
    """Clear all context variables."""
    if _STRUCTLOG_AVAILABLE:
        structlog.contextvars.clear_contextvars()


class LogContext:
    """Context manager for temporarily setting log context."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.previous: dict[str, Any] | None = None

    def __enter__(self) -> None:
        if _STRUCTLOG_AVAILABLE:
            self.previous = structlog.contextvars.get_merged_contextvars()
            structlog.contextvars.bind_contextvars(**self.kwargs)

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if _STRUCTLOG_AVAILABLE:
            structlog.contextvars.clear_contextvars()
            if self.previous:
                structlog.contextvars.bind_contextvars(**self.previous)


# Convenience functions for common log events
def log_task_start(logger: Any, task_id: str, description: str) -> None:
    """Log task start event."""
    logger.info("task_started", task_id=task_id, description=description)


def log_task_complete(logger: Any, task_id: str, success: bool, duration: float | None = None) -> None:
    """Log task completion event."""
    logger.info("task_completed", task_id=task_id, success=success, duration=duration)


def log_cycle_start(logger: Any, cycle: int, goal: str) -> None:
    """Log cycle start event."""
    logger.info("cycle_started", cycle=cycle, goal=goal)


def log_cycle_complete(logger: Any, cycle: int, tests_passed: bool, duration: float | None = None) -> None:
    """Log cycle completion event."""
    logger.info("cycle_completed", cycle=cycle, tests_passed=tests_passed, duration=duration)


def log_llm_call(
    logger: Any,
    model: str,
    provider: str,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    duration: float | None = None,
) -> None:
    """Log LLM call event."""
    logger.info(
        "llm_call",
        model=model,
        provider=provider,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=cost,
        duration=duration,
    )


def log_error(logger: Any, error: Exception, context: dict[str, Any] | None = None) -> None:
    """Log error event with structured context."""
    logger.error(
        "error",
        error_type=type(error).__name__,
        error_message=str(error),
        context=context or {},
        exc_info=True,
    )