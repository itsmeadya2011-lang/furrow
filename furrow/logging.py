from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with sensible defaults for Console + JSON output."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stderr,
        format="%(message)s",
    )
    shared_processors: list[Any] = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]
    if sys.stderr.isatty():
        processors = shared_processors + [
            structlog.processors.ConsoleRenderer(),
        ]
    else:
        processors = shared_processors + [
            structlog.processors.JSONRenderer(),
        ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    configure_logging()
    if name:
        return structlog.get_logger(name)
    return structlog.get_logger()
