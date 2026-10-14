from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.dev import ConsoleRenderer


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog with console renderer, timestamps, and level coloring."""
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            ConsoleRenderer(colors=True),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a structlog logger bound to the given name."""
    return structlog.get_logger(name)