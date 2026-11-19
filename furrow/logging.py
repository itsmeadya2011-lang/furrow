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
    structlog = None  # type: ignore[assignment]
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


def bind_context(logger: Any, **kwargs: Any) -> Any:
    """Bind context variables to a logger for structured logging.
    
    This allows adding contextual information (like run_id, task_id, cycle, etc.)
    to all log messages from that logger.
    """
    if _STRUCTLOG_AVAILABLE and hasattr(logger, "bind"):
        return logger.bind(**kwargs)
    return logger


class LoggingContext:
    """Context manager for adding structured logging context.
    
    Usage:
        with LoggingContext(run_id="abc123", cycle=1):
            logger.info("Starting cycle")
    """
    
    def __init__(self, logger: Any, **kwargs: Any) -> None:
        self.logger = logger
        self.kwargs = kwargs
        self._bound_logger = None
    
    def __enter__(self) -> Any:
        if _STRUCTLOG_AVAILABLE and hasattr(self.logger, "bind"):
            self._bound_logger = self.logger.bind(**self.kwargs)
            return self._bound_logger
        return self.logger
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        # Nothing to clean up for structlog
        pass


def log_with_context(logger: Any, level: str, message: str, **kwargs: Any) -> None:
    """Log a message with structured context.
    
    This is a helper for stdlib loggers to add context in a consistent way.
    For structlog loggers, use logger.bind() directly.
    """
    if _STRUCTLOG_AVAILABLE and hasattr(logger, "bind"):
        bound = logger.bind(**kwargs)
        getattr(bound, level.lower())(message)
    else:
        # For stdlib logging, add context to the message
        context_str = " ".join(f"{k}={v}" for k, v in kwargs.items())
        full_message = f"{message} [{context_str}]" if context_str else message
        getattr(logger, level.lower())(full_message)