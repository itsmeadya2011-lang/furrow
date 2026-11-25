"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
"""

from __future__ import annotations

import logging
import os
import sys
from enum import Enum
from typing import Any


class LogFormat(str, Enum):
    """Supported log output formats."""

    JSON = "json"
    CONSOLE = "console"


def _configure_stdlib_fallback(level: str = "INFO", fmt: LogFormat = LogFormat.CONSOLE) -> None:
    """Configure standard library logging as a fallback."""
    if fmt == LogFormat.JSON:
        # Use a simple JSON-like format for stdlib
        log_format = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    else:
        log_format = "%(asctime)s %(levelname)s %(name)s %(message)s"
    logging.basicConfig(
        format=log_format,
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore[assignment]
    _STRUCTLOG_AVAILABLE = False


def configure_logging(level: str = "INFO", fmt: str | LogFormat | None = None) -> None:
    """Configure structlog to wrap stdlib logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        fmt: Log format - "json" for structured JSON, "console" for human-readable.
             Defaults to JSON when structlog is available, otherwise console.
             Can also be set via FURROW_LOG_FORMAT environment variable.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Determine log format
    if fmt is None:
        fmt = os.getenv("FURROW_LOG_FORMAT", "json" if _STRUCTLOG_AVAILABLE else "console")
    if isinstance(fmt, str):
        fmt = LogFormat(fmt.lower())

    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    if not _STRUCTLOG_AVAILABLE:
        # Fall back to stdlib logging if structlog is not available
        _configure_stdlib_fallback(level, fmt)
        return

    if fmt == LogFormat.JSON:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ]
    else:
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a structlog logger, falling back to stdlib if needed."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)
