from __future__ import annotations

import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from furrow.config import Settings, settings


def add_log_level(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Add log level to event dict."""
    event_dict["level"] = method_name.upper()
    return event_dict


def drop_debug_fields(logger: Any, method_name: str, event_dict: EventDict) -> EventDict:
    """Remove debug fields from production logs."""
    if method_name in ("debug",):
        return event_dict
    # Remove potentially sensitive fields in non-debug logs
    for key in list(event_dict.keys()):
        if key.startswith("_"):
            event_dict.pop(key, None)
    return event_dict


def setup_logging(settings: Settings = settings) -> None:
    """Configure structured logging for Furrow."""
    log_level = settings.log_level.upper()
    
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
    ]
    
    if sys.stderr.isatty():
        # Pretty console output for development
        structlog.configure(
            processors=shared_processors + [
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )
    else:
        # JSON output for production/logging aggregation
        structlog.configure(
            processors=shared_processors + [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)