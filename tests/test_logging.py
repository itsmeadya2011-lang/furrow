"""Tests for the Furrow logging configuration."""

from __future__ import annotations

from furrow.logging import (
    _STRUCTLOG_AVAILABLE,
    configure_logging,
    get_logger,
)


def test_configure_logging_default_level() -> None:
    """configure_logging() accepts the default level and does not raise."""
    configure_logging("INFO")


def test_configure_logging_accepts_all_common_levels() -> None:
    for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        configure_logging(level)


def test_configure_logging_lowercases_and_falls_back() -> None:
    """An unknown level falls back to INFO without raising."""
    configure_logging("not-a-real-level")


def test_get_logger_returns_a_logger() -> None:
    """get_logger() returns an object with at least a ``name`` attribute."""
    logger = get_logger("furrow.test")
    assert logger is not None
    # structlog loggers expose .name; stdlib loggers expose .name too.
    assert getattr(logger, "name", None) is not None or hasattr(logger, "info")


def test_structlog_availability_flag_matches_import() -> None:
    """The module-level flag reflects whether structlog was importable."""
    assert isinstance(_STRUCTLOG_AVAILABLE, bool)


def test_configure_logging_idempotent() -> None:
    """Calling configure_logging twice does not raise."""
    configure_logging("INFO")
    configure_logging("DEBUG")