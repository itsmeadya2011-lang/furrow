"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def _configure_stdlib_fallback(level: str = "INFO") -> None:
    """Configure standard library logging as a fallback."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None
    _STRUCTLOG_AVAILABLE = False


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to wrap stdlib logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    if not _STRUCTLOG_AVAILABLE:
        # Fall back to stdlib logging if structlog is not available
        _configure_stdlib_fallback(level)
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
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
