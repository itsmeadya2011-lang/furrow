"""Logging configuration for Furrow using structlog."""
from __future__ import annotations

import logging
import sys

import structlog
from structlog.dev import ConsoleRenderer


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog and the stdlib logging facade.

    Safe to call multiple times — it clears existing handlers first.
    """
    level_enum = getattr(logging, level.upper(), logging.INFO)

    # Reset stdlib logging handlers so we don't accumulate output on re-entry.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level_enum,
        format="%(message)s",
        stream=sys.stderr,
    )

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_enum),
        cache_logger_on_first_use=True,
    )
