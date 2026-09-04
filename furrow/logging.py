from __future__ import annotations

import logging

import structlog

from furrow.config import settings

logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(logging, settings.log_level.upper(), logging.INFO)
    ),
    cache_logger_on_first=True,
)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Return a configured structlog logger."""
    return structlog.get_logger(name)
