from __future__ import annotations

import logging
import sys

import structlog

_FORMATTER = {
    "processors": [
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ColorizedStderrRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
    ],
    "wrapper_class": structlog.make_filtering_bound_logger(logging.INFO),
    "cache_logger_on_first_use": True,
}


def configure_logging(log_level: str = "INFO") -> None:
    """Configure root logging and structlog.

    Safe to call multiple times; idempotent.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    structlog.configure(
        processors=_FORMATTER["processors"],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "__main__"):
    return structlog.get_logger(name)
