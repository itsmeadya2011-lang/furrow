"""Logging configuration for Furrow using structlog."""
from __future__ import annotations

import logging
import sys

import structlog

from furrow.config import Settings


def configure_logging(settings: Settings | None = None) -> logging.Logger:
    """Configure structlog for console output.

    Call this early in :func:`furrow.cli.main.main` to enable structured
    logging throughout the application.
    """
    s = settings or Settings()
    level = getattr(logging, s.log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
    )

    if not structlog.is_configured():
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(format="iso"),  # type: ignore[call-arg]
                structlog.processors.add_log_level,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_wrapper_class(level),  # type: ignore[arg-type, operator]
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )

    return structlog.get_logger("furrow")  # type: ignore[no-any-return]
