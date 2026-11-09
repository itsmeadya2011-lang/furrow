"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.

Supports correlation IDs and structured context for better debugging.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from furrow.structured_logging import (
    configure_logging_with_correlation,
    get_correlation_id,
    get_structured_context,
    merge_structured_context,
    set_correlation_id,
    set_structured_context,
)


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
    structlog = None  # type: ignore[assignment]
    _STRUCTLOG_AVAILABLE = False


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to wrap stdlib logging with correlation ID support."""
    configure_logging_with_correlation(level)


def get_logger(name: str) -> Any:
    """Return a structlog logger, falling back to stdlib if needed."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)


# Re-export structured logging utilities
__all__ = [
    "configure_logging",
    "get_logger",
    "set_correlation_id",
    "get_correlation_id",
    "set_structured_context",
    "get_structured_context",
    "merge_structured_context",
]
