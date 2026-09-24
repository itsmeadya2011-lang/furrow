from __future__ import annotations

import sys
from pathlib import Path

import structlog
from structlog.stdlib import LoggerFactory

from furrow.config import settings


def setup_logging(log_level: str | None = None, log_file: Path | None = None) -> None:
    """Configure structlog for the application."""
    level = log_level or settings.log_level
    
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if log_file:
        # File output with JSON
        processors.append(structlog.processors.JSONRenderer())
        file_handler = structlog.WriteLoggerFactory(log_file.open("a"))
    else:
        # Console output with colors
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
        file_handler = LoggerFactory()
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=file_handler,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a logger instance."""
    return structlog.get_logger(name)