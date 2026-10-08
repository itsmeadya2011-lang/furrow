from __future__ import annotations

import logging

import structlog

from furrow.config import settings

__all__ = ["logger", "configure"]


def configure() -> structlog.BoundLogger:
    level = getattr(logging, str(settings.log_level).upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        level=level,
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.WriteLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("furrow")


logger = configure()
