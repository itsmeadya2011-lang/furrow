from __future__ import annotations

import logging

import structlog


def configure_logging(log_level: str = "INFO") -> structlog.BoundLogger:
    """Configure structlog with a ConsoleRenderer and return a root logger."""
    level = log_level.upper()
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    return structlog.get_logger()


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Return a structlog logger, configuring at the settings log_level if not yet done."""
    return structlog.get_logger(name)


logger: structlog.BoundLogger = configure_logging("INFO")
