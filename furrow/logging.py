"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.

Supports multiple output formats:
- JSON: structured output for log aggregation
- Console: human-readable output for development
- Plain: simple output for minimal environments
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Literal


LogFormat = Literal["json", "console", "plain"]


def _configure_stdlib_fallback(level: str = "INFO", fmt: LogFormat = "plain") -> None:
    """Configure standard library logging as a fallback."""
    if fmt == "json":
        # JSON-like format for stdlib
        format_str = '{"time": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
    elif fmt == "console":
        format_str = "%(asctime)s %(levelname)-8s %(name)s %(message)s"
    else:
        format_str = "%(asctime)s %(levelname)s %(name)s %(message)s"
    
    logging.basicConfig(
        format=format_str,
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore[assignment]
    _STRUCTLOG_AVAILABLE = False


def configure_logging(
    level: str = "INFO",
    fmt: LogFormat = "console",
    add_context: bool = True,
) -> None:
    """Configure structlog to wrap stdlib logging.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        fmt: Output format - "json" for structured logs, "console" for human-readable,
             "plain" for simple output
        add_context: Whether to include context variables in log output
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Configure stdlib logging as the base
    if fmt == "json":
        logging.basicConfig(
            format="%(message)s",
            level=log_level,
            stream=sys.stdout,
        )
    elif fmt == "console":
        logging.basicConfig(
            format="%(message)s",
            level=log_level,
            stream=sys.stdout,
        )
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            level=log_level,
            stream=sys.stdout,
        )

    if not _STRUCTLOG_AVAILABLE:
        # Fall back to stdlib logging if structlog is not available
        _configure_stdlib_fallback(level, fmt)
        return

    # Build processor chain based on format
    processors = [
        structlog.contextvars.merge_contextvars if add_context else lambda _, __, event_dict: event_dict,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    
    if fmt == "json":
        processors.extend([
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ])
    elif fmt == "console":
        processors.extend([
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True),
        ])
    else:  # plain
        processors.extend([
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.KeyValueRenderer(key_order=["timestamp", "level", "logger", "event"]),
        ])

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


def configure_logging_from_env() -> None:
    """Configure logging from environment variables.
    
    Environment variables:
        FURROW_LOG_LEVEL: Log level (default: INFO)
        FURROW_LOG_FORMAT: Log format - json, console, plain (default: console)
        FURROW_LOG_CONTEXT: Whether to add context vars - true/false (default: true)
    """
    level = os.getenv("FURROW_LOG_LEVEL", "INFO")
    fmt = os.getenv("FURROW_LOG_FORMAT", "console")
    add_context = os.getenv("FURROW_LOG_CONTEXT", "true").lower() == "true"
    
    # Validate format
    if fmt not in ("json", "console", "plain"):
        fmt = "console"
    
    configure_logging(level=level, fmt=fmt, add_context=add_context)
