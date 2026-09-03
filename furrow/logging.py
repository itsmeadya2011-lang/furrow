from __future__ import annotations

import logging
import sys

import structlog

_DEFAULT_LEVEL = "INFO"


def _resolve_level(level: str) -> int:
    """Translate a human level name into a numeric stdlib level."""
    name = level.upper().strip()
    numeric = logging.getLevelName(name)
    if isinstance(numeric, int):
        return numeric
    # Fall back to INFO when the level is unrecognised.
    return logging.getLevelName(_DEFAULT_LEVEL)


def configure_logging(level: str = _DEFAULT_LEVEL, *, json: bool | None = None) -> None:
    """Configure structlog and the stdlib ``logging`` root logger.

    Parameters
    ----------
    level:
        Desired minimum log level (case-insensitive). Anything emitted below
        this level is dropped by both structlog and the root handler.
    json:
        Force a specific renderer. When ``None`` (the default) we use the
        pretty ``ConsoleRenderer`` if ``stdout`` is a TTY, otherwise JSON for
        machine-friendly output.
    """
    numeric_level = _resolve_level(level)

    # Route stdlib logging through structlog so anything that uses
    # ``logging.getLogger(...)`` is rendered consistently.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=numeric_level,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.stdlib.add_logger_name,
    ]

    if json is True:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    elif json is False:
        renderer = structlog.dev.ConsoleRenderer()
    else:
        renderer = (
            structlog.dev.ConsoleRenderer()
            if sys.stdout.isatty()
            else structlog.processors.JSONRenderer()
        )

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(numeric_level)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    # Replace any existing handlers so our formatter is authoritative.
    root.handlers = [handler]


class LogLevelFilter(logging.Filter):
    """Stdlib ``logging`` filter that enforces a minimum level.

    Useful when attaching structlog's formatter to a handler whose own level
    is set higher than desired for certain loggers.
    """

    def __init__(self, level: int | str = _DEFAULT_LEVEL) -> None:
        super().__init__()
        if isinstance(level, str):
            level = _resolve_level(level)
        self._level = level

    @property
    def level(self) -> int:
        return self._level

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: A003
        return record.levelno >= self._level


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structlog bound logger."""
    if name is None:
        return structlog.get_logger()
    return structlog.get_logger(name)


# Lazy-initialised module-level logger. Configuration is applied lazily so
# importing this module is side-effect free.
logger: structlog.stdlib.BoundLogger = structlog.get_logger("furrow")


__all__ = [
    "configure_logging",
    "get_logger",
    "logger",
    "LogLevelFilter",
]