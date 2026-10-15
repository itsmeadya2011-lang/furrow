from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from furrow.config import settings

# Map string log levels to numeric values
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _configure_logger() -> structlog.BoundLogger:
    """Configure and return a structlog logger with rich-friendly console output."""
    level = _LEVEL_MAP.get(settings.log_level.upper(), logging.INFO)

    # Clear any existing handlers to avoid duplicate output
    root = logging.getLogger()
    root.handlers = []
    root.setLevel(level)

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if level <= logging.INFO else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.WriteLoggerFactory(sys.stderr),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger("furrow")


logger: structlog.BoundLogger = _configure_logger()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Return a structlog logger bound with an optional name context."""
    if name:
        # Bind and return the bound instance (not lazy proxy)
        return logger.bind(component=name)
    # Force binding to get actual instance for v26 compatibility
    return logger.bind()


def log_event(level: str, event: str, **kwargs: Any) -> None:
    """Convenience function to log an event at the given level."""
    level_lower = level.lower() if level else "info"
    try:
        log_method = getattr(logger, level_lower)
        log_method(event, **kwargs)
    except (AttributeError, Exception):
        logger.info(event, **kwargs)
