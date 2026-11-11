"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.

Provides context variables for cycle, task_id, and other run metadata that
are automatically included in all log entries when bound.
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from typing import Any

import structlog

# Context variables for structured logging
cycle_var: ContextVar[int | None] = ContextVar("furrow_cycle", default=None)
task_id_var: ContextVar[str | None] = ContextVar("furrow_task_id", default=None)
goal_var: ContextVar[str | None] = ContextVar("furrow_goal", default=None)
phase_var: ContextVar[str | None] = ContextVar("furrow_phase", default=None)  # plan, execute, test


def _configure_stdlib_fallback(level: str = "INFO") -> None:
    """Configure standard library logging as a fallback."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


def _add_contextvars(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Add context variables to the log event."""
    cycle = cycle_var.get()
    if cycle is not None:
        event_dict["cycle"] = cycle
    task_id = task_id_var.get()
    if task_id is not None:
        event_dict["task_id"] = task_id
    goal = goal_var.get()
    if goal is not None:
        event_dict["goal"] = goal[:100] if len(goal) > 100 else goal
    phase = phase_var.get()
    if phase is not None:
        event_dict["phase"] = phase
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
            _add_contextvars,
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


def bind_cycle(cycle: int) -> None:
    """Bind the current cycle number to context variables."""
    cycle_var.set(cycle)


def bind_task_id(task_id: str) -> None:
    """Bind the current task ID to context variables."""
    task_id_var.set(task_id)


def bind_goal(goal: str) -> None:
    """Bind the current goal to context variables."""
    goal_var.set(goal)


def bind_phase(phase: str) -> None:
    """Bind the current phase (plan, execute, test) to context variables."""
    phase_var.set(phase)


def clear_context() -> None:
    """Clear all context variables."""
    cycle_var.set(None)
    task_id_var.set(None)
    goal_var.set(None)
    phase_var.set(None)
