from __future__ import annotations

import sys
from typing import Any

import structlog
from structlog.types import Processor


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog for structured logging."""
    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "furrow") -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)