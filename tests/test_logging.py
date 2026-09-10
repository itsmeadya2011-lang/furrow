import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from furrow.logging import configure_logging, get_logger


def test_configure_logging():
    """Test that logging configuration doesn't raise."""
    configure_logging()
    # If we get here without exception, it works


def test_get_logger():
    """Test getting a logger instance."""
    logger = get_logger("test.module")
    assert logger is not None
    # Test that we can call log methods
    logger.info("test message", key="value")
    logger.debug("debug message")
    logger.warning("warning message")
    logger.error("error message")


def test_structlog_output(capsys):
    """Test that structlog produces JSON output."""
    configure_logging()
    logger = get_logger("test.output")
    logger.info("test event", foo="bar")
    captured = capsys.readouterr()
    assert "test event" in captured.out
    assert "foo" in captured.out
    assert "bar" in captured.out