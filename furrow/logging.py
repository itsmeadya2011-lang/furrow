from __future__ import annotations

import logging
import sys

try:
    import structlog
except ImportError:  # pragma: no cover - structlog is a declared dependency
    structlog = None  # type: ignore[assignment]


def configure(log_level: str = "INFO") -> None:
    """Configure structured logging (stdlib + structlog if available).

    Safe to call multiple times. Does not mutate log level after first call
    because ``logging.basicConfig`` is a no-op once configured.
    """
    level = logging.getLevelName(log_level.upper())
    if not isinstance(level, int):
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stderr,
    )

    if structlog is not None:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str | None = None):
    """Return a logger exposing ``.info/.debug/.warning/.error``.

    Uses structlog when available, falling back to the stdlib logging module.
    """
    if structlog is not None:
        return structlog.get_logger(name)
    return logging.getLogger(name)
