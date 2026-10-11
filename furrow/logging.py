from __future__ import annotations

import sys
import types
from typing import Any

import structlog

from furrow.config import Settings


def setup_logging(settings: Settings | None = None) -> None:
    """Configure structlog with rich console output."""
    s = settings or Settings()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Set log level
    import logging

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, s.log_level.upper(), logging.INFO),
    )


def get_logger(name: str = "furrow") -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class LogContext:
    """Context manager for adding structured context to logs."""

    def __init__(self, logger: structlog.BoundLogger, **kwargs: Any) -> None:
        self.logger = logger
        self.kwargs = kwargs
        self.token = None

    def __enter__(self) -> structlog.BoundLogger:
        self.token = structlog.contextvars.bind_contextvars(**self.kwargs)
        return self.logger

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        if self.token:
            structlog.contextvars.unbind_contextvars(*self.kwargs.keys())


# Convenience function for use in agents
def log_task_start(logger: structlog.BoundLogger, task_id: str, description: str) -> None:
    logger.info("task_started", task_id=task_id, description=description)


def log_task_complete(logger: structlog.BoundLogger, task_id: str, result: str) -> None:
    logger.info("task_completed", task_id=task_id, result=result[:200])


def log_task_failed(logger: structlog.BoundLogger, task_id: str, error: str) -> None:
    logger.error("task_failed", task_id=task_id, error=error)


def log_cycle_start(logger: structlog.BoundLogger, cycle: int, goal: str) -> None:
    logger.info("cycle_started", cycle=cycle, goal=goal[:100])


def log_cycle_complete(
    logger: structlog.BoundLogger, cycle: int, tasks_completed: int, tests_passed: bool
) -> None:
    logger.info(
        "cycle_completed", cycle=cycle, tasks_completed=tasks_completed, tests_passed=tests_passed
    )
