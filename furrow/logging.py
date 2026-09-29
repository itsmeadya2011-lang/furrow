"""Structured logging configuration for Furrow using structlog."""

from __future__ import annotations

import logging
import sys

import structlog
from structlog.stdlib import ProcessorFormatter

from furrow.config import settings

_LOG_FORMATTER = ProcessorFormatter(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer() if settings.log_level == "DEBUG" else structlog.processors.ConsoleRenderer(),
    ],
    foreign_pre_chain=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
    ],
    wrapper_class=structlog.make_wrapper_class(logging.Logger, passthrough_2_args=True),
)

_CONFIGURED = False


def configure_logging() -> structlog.BoundLogger:
    """Configure structlog and standard logging. Returns the root logger."""
    global _CONFIGURED
    if _CONFIGURED:
        return structlog.get_logger("furrow")

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stderr,
        handlers=[logging.StreamHandler(sys.stderr)],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_LOG_FORMATTER)
    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    structlog.configure(
        processors=_LOG_FORMATTER.pre_chain,
        wrapper_class=structlog.make_wrapper_class(logging.Logger, passthrough_2_args=True),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first=True,
    )

    _CONFIGURED = True
    return structlog.get_logger("furrow")


def get_logger(name: str = "furrow") -> structlog.BoundLogger:
    """Get a structlog logger, configuring logging on first use."""
    if not _CONFIGURED:
        configure_logging()
    return structlog.get_logger(name)
