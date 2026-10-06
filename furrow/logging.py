from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog
from structlog.types import EventDict, WrappedLogger

from furrow.config import settings


def add_log_level(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def add_timestamp(logger: WrappedLogger, method_name: str, event_dict: EventDict) -> EventDict:
    """Add timestamp to event dict."""
    import time
    event_dict["timestamp"] = time.time()
    return event_dict


def setup_logging(log_level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure structlog for the application."""
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level_int,
    )

    processors = [
        structlog.contextvars.merge_contextvars,
        add_log_level,
        add_timestamp,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    if sys.stderr.isatty():
        # Pretty console output for TTY
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        # JSON output for non-TTY (e.g., Docker, CI)
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level_int),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Also configure file logging if requested
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level_int)
        file_handler.setFormatter(logging.Formatter("%(message)s"))
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Get a structlog logger."""
    return structlog.get_logger(name)


# Initialize logging on import
setup_logging(settings.log_level)