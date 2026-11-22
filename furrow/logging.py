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
        _configure_stdlib_fallback(level)
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
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


def log_llm_call(
    logger: Any,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost: float,
    latency_ms: float,
) -> None:
    """Log an LLM call with structured fields."""
    logger.info(
        "llm_call",
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cost_usd=cost,
        latency_ms=latency_ms,
    )


def log_task_event(
    logger: Any,
    task_id: str,
    event: str,
    status: str = "",
    attempts: int = 0,
    duration_ms: float = 0,
    error: str = "",
) -> None:
    """Log a task lifecycle event."""
    logger.info(
        "task_event",
        task_id=task_id,
        event=event,
        status=status,
        attempts=attempts,
        duration_ms=duration_ms,
        error=error,
    )


def log_cycle_event(
    logger: Any,
    cycle: int,
    event: str,
    tasks_planned: int = 0,
    tasks_completed: int = 0,
    tests_passed: bool | None = None,
) -> None:
    """Log an orchestration cycle event."""
    logger.info(
        "cycle_event",
        cycle=cycle,
        event=event,
        tasks_planned=tasks_planned,
        tasks_completed=tasks_completed,
        tests_passed=tests_passed,
    )


def log_tool_call(
    logger: Any,
    task_id: str,
    tool_name: str,
    success: bool,
    duration_ms: float,
    error: str = "",
) -> None:
    """Log a tool call."""
    logger.info(
        "tool_call",
        task_id=task_id,
        tool=tool_name,
        success=success,
        duration_ms=duration_ms,
        error=error,
    )