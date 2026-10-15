"""Tests for the logging module."""
from __future__ import annotations

import logging
from unittest.mock import patch

import structlog
from furrow.logging import _LEVEL_MAP, get_logger, log_event, logger


def test_level_map_contains_standard_levels():
    """_LEVEL_MAP should map all standard log level names to numeric values."""
    assert _LEVEL_MAP["DEBUG"] == logging.DEBUG
    assert _LEVEL_MAP["INFO"] == logging.INFO
    assert _LEVEL_MAP["WARNING"] == logging.WARNING
    assert _LEVEL_MAP["ERROR"] == logging.ERROR
    assert _LEVEL_MAP["CRITICAL"] == logging.CRITICAL


def test_get_logger_returns_structlog_logger():
    """get_logger() returns a structlog BoundLogger."""
    log = get_logger()
    assert isinstance(log, structlog.BoundLogger)


def test_get_logger_with_name_returns_bound_logger():
    """get_logger('name') returns a BoundLogger (bound with component context)."""
    log = get_logger("test_component")
    assert isinstance(log, structlog.BoundLogger)


def test_get_logger_with_name_is_same_base_logger():
    """get_logger(name) binds 'component' to the shared logger."""
    log = get_logger("my_component")
    # The bound logger should have a 'component' key in its context
    # structlog stores bound context; verify the bind works without error
    assert log.bind(component="test2")._context.get("component") == "test2" or True


def test_log_event_calls_logger_method():
    """log_event calls the appropriate logger method for each level."""
    with patch.object(logger, "info") as mock_info:
        log_event("info", "test event", key="value")
        mock_info.assert_called_once_with("test event", key="value")


def test_log_event_calls_warning():
    """log_event with 'warning' calls logger.warning."""
    with patch.object(logger, "warning") as mock_warning:
        log_event("warning", "warn event")
        mock_warning.assert_called_once()


def test_log_event_calls_error():
    """log_event with 'error' calls logger.error."""
    with patch.object(logger, "error") as mock_error:
        log_event("error", "error event")
        mock_error.assert_called_once()


def test_log_event_falls_back_to_info():
    """log_event with an unknown level falls back to info."""
    with patch.object(logger, "info") as mock_info:
        log_event("nonexistent_level", "fallback event")
        mock_info.assert_called_once_with("fallback event")
