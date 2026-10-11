from __future__ import annotations

import logging
import sys

import structlog

from furrow.config import settings

LOG_LEVEL = getattr(logging, settings.log_level.upper(), logging.INFO)

logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=LOG_LEVEL,
)

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(LOG_LEVEL),
    logger_factory=structlog.WriteLoggerFactory(file=sys.stdout),
    cache_logger_on_first_use=True,
)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    return structlog.get_logger(name)
