from __future__ import annotations

import sys
from typing import Any

import structlog
from structlog.types import FilteringBoundLogger

from furrow.config import Settings, settings


def configure_logging(config: Settings = settings) -> None:
    """Configure structured logging for Furrow."""
    log_level = config.log_level.upper()
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(colors=True) if sys.stderr.isatty() 
            else structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    import logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, log_level, logging.INFO),
    )


def get_logger(name: str = "furrow") -> FilteringBoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)