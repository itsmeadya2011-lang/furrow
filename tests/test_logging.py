"""Tests for the logging module (furrow.logging)."""

from __future__ import annotations

from furrow.logging import _STRUCTLOG_AVAILABLE, configure_logging, get_logger


def test_configure_logging_does_not_raise():
    """configure_logging should complete without raising for valid levels."""
    for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
        configure_logging(level)


def test_configure_logging_invalid_level_does_not_raise():
    """An unknown level name should fall back gracefully instead of crashing."""
    configure_logging("NOT_A_LEVEL")


def test_get_logger_returns_logger():
    """get_logger should return a logger object (structlog or stdlib)."""
    logger = get_logger("furrow.test.logging.get_logger")
    assert logger is not None
    assert hasattr(logger, "debug") or hasattr(logger, "info")


def test_structlog_available_flag_is_bool():
    """_STRUCTLOG_AVAILABLE should be a boolean reflecting import success."""
    assert isinstance(_STRUCTLOG_AVAILABLE, bool)


def test_get_logger_does_not_raise(caplog):
    """Getting a logger and emitting through it should not raise."""
    configure_logging("INFO")
    logger = get_logger("furrow.test.logging.no_raise")
    logger.info("test message")
    logger.debug("debug message")
    logger.warning("warning message")


def test_configure_logging_idempotent():
    """Calling configure_logging multiple times should not raise."""
    configure_logging("INFO")
    configure_logging("DEBUG")
    configure_logging("WARNING")
