"""Structured logging for the Furrow CLI entry points.

Logs go to **stderr**, always.  The user-facing progress in the orchestrator and
the CLI is ``rich`` console output on stdout, so a log line on stdout would
corrupt the terminal rendering and break piping (``furrow start ... | tee``).
Keeping the two streams apart is the whole point of this module.

Importing ``furrow.logging_config`` has no side effects: it never calls
:func:`configure_logging` for you.  Configuring logging is a side effect on the
*host* process, so a library import must not do it; the CLI calls
:func:`configure_logging` once at startup instead.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

#: Level used when ``Settings.log_level`` is missing or unrecognised.  Anything
#: louder would spam a terminal; anything quieter would hide loop failures.
DEFAULT_LEVEL = "INFO"

#: Loggers created before :func:`configure_logging` runs would otherwise be
#: frozen at structlog's default (WARNING).  Reconfiguring replaces the
#: configuration globally, so previously handed-out loggers follow the new one
#: as long as they are not cached with ``cache_logger_on_first_use``.
_CACHE_LOGGERS_ON_FIRST_USE = False


def resolve_level(level: str | int | None) -> int:
    """Return the numeric stdlib level for ``level``, degrading to INFO.

    Accepts the standard names (``DEBUG``, ``INFO``, ``WARNING``/``WARN``,
    ``ERROR``, ``CRITICAL``/``FATAL``), ``NOTSET``, and bare numbers.  An
    unrecognised value is not fatal: a typo in ``FURROW_LOG_LEVEL`` must not
    stop the agent from running, so it falls back to :data:`DEFAULT_LEVEL`.
    """
    if isinstance(level, bool):  # bool is an int subclass; never a level
        return logging.INFO
    if isinstance(level, int):
        return level
    if not isinstance(level, str):
        return logging.INFO
    name = level.strip().upper()
    named = getattr(logging, name, None)
    if isinstance(named, int):
        return named
    if name.lstrip("-").isdigit():
        return int(name)
    return logging.INFO


def configure_logging(level: str | int | None = DEFAULT_LEVEL) -> None:
    """Route structured logs to stderr at ``level``.

    Uses the stdlib ``logging`` machinery under the hood so that one
    configuration covers both ``structlog`` loggers and the plain stdlib
    loggers other modules obtain with ``logging.getLogger``.  ``force=True``
    because an entry point owns the process: handlers installed by an importer
    would otherwise capture the output away from stderr.  Safe to call more
    than once.
    """
    numeric = resolve_level(level)

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=numeric, force=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(numeric),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=_CACHE_LOGGERS_ON_FIRST_USE,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger for ``name``.

    Usable before :func:`configure_logging`: structlog's default configuration
    is harmless (it prints to stdout), it just is not the one we want.
    """
    return structlog.get_logger(name)
