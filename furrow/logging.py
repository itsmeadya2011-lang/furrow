"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.

Supports both human-readable (console) and structured (JSON) output formats.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def _configure_stdlib_fallback(level: str = "INFO", json_output: bool = False) -> None:
    """Configure standard library logging as a fallback."""
    if json_output:
        # Use JSON format for stdlib fallback
        import json as json_module

        class JsonFormatter(logging.Formatter):
            def format(self, record: logging.LogRecord) -> str:
                log_data = {
                    "timestamp": self.formatTime(record, self.datefmt),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                if record.exc_info:
                    log_data["exception"] = self.formatException(record.exc_info)
                return json_module.dumps(log_data)

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            handlers=[handler],
        )
    else:
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


def configure_logging(
    level: str = "INFO",
    json_output: bool = False,
    include_timestamp: bool = True,
) -> None:
    """Configure structlog to wrap stdlib logging.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_output: If True, output structured JSON logs instead of human-readable
        include_timestamp: If True, include ISO timestamp in logs
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Configure stdlib logging to pass through to structlog
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    if not _STRUCTLOG_AVAILABLE:
        # Fall back to stdlib logging if structlog is not available
        _configure_stdlib_fallback(level, json_output)
        return

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if include_timestamp:
        processors.insert(1, structlog.processors.TimeStamper(fmt="iso"))

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Human-readable console output with colors
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

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