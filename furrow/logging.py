from __future__ import annotations

import logging
import sys

import structlog
from structlog.stdlib import LoggerFactory

from furrow.config import Settings


def _configure_logging(settings: Settings | None = None) -> None:
    settings = settings or Settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    processors = [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]

    structlog.configure(
        processors=processors,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
            ],
            processors=processors,
        )
    )
    logging.basicConfig(level=log_level, handlers=[handler], force=True)


_configure_logging()


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
