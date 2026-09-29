from __future__ import annotations

import logging
import sys

import structlog
from structlog.dev import ConsoleRenderer


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog and the standard logging module.

    Structlog is configured with ``add_log_level``, an ISO ``TimeStamper`` and a
    renderer (``ConsoleRenderer`` while debugging, ``JSONRenderer`` otherwise).
    The standard library ``logging`` module is also configured so that
    third-party loggers such as uvicorn and fastapi have their records captured
    at the requested level.

    ``log_level`` is expected to come from :class:`furrow.config.Settings` (the
    CLI passes ``settings.log_level``), defaulting to ``"INFO"`` when called
    directly.
    """
    level = getattr(logging, str(log_level).upper(), logging.INFO)

    renderer: structlog.types.Processor = (
        ConsoleRenderer()
        if level == logging.DEBUG
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )

    # Configure the standard logging module so third-party loggers
    # (uvicorn, fastapi, ...) are captured at the configured level.
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
        force=True,
    )
