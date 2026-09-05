"""Logging configuration for Furrow using structlog."""
from __future__ import annotations

import logging
import sys

import structlog

_LOG_CONFIGURED = False


def configure(level: str = "INFO") -> None:
    """Configure structlog logging. Safe to call multiple times."""
    global _LOG_CONFIGURED
    if _LOG_CONFIGURED:
        return
    _LOG_CONFIGURED = True

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        level=numeric_level,
        format="%(message)s",
        stream=sys.stderr,
    )

    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(colors=False),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.WriteLoggerFactory(file=sys.stderr),
        cache_logger_on_start=True,
    )


def get_logger(name: str = "furrow"):
    """Get a structlog logger, configuring defaults if needed."""
    if not _LOG_CONFIGURED:
        configure()
    return structlog.get_logger(name)
