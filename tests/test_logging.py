"""Tests for the Furrow logging module."""

from __future__ import annotations

import logging

import pytest

from furrow.logging import (
    _STRUCTLOG_AVAILABLE,
    _configure_stdlib_fallback,
    configure_logging,
    get_logger,
)


@pytest.fixture(autouse=True)
def _reset_logging_state():
    """Save and restore root logger state so tests do not leak configuration."""
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    yield
    root.setLevel(saved_level)
    root.handlers[:] = saved_handlers


def test_structlog_availability_flag_is_bool():
    assert isinstance(_STRUCTLOG_AVAILABLE, bool)


def test_configure_logging_with_default_level_does_not_raise():
    """configure_logging accepts the default level and does not raise."""
    configure_logging()


def test_configure_logging_with_debug_level_does_not_raise():
    configure_logging("DEBUG")


def test_configure_logging_with_invalid_level_does_not_raise():
    """An unknown level name falls back to INFO and does not raise."""
    configure_logging("NOT_A_LEVEL")


def test_configure_logging_idempotent():
    """Calling configure_logging multiple times does not raise."""
    configure_logging("INFO")
    configure_logging("WARNING")
    configure_logging("ERROR")


def test_get_logger_returns_a_logger():
    logger = get_logger("furrow.test")
    assert logger is not None


def test_get_logger_with_empty_name():
    logger = get_logger("")
    assert logger is not None


def test_configure_stdlib_fallback_does_not_raise():
    _configure_stdlib_fallback("INFO")


def test_configure_stdlib_fallback_with_debug():
    _configure_stdlib_fallback("DEBUG")


def test_configure_stdlib_fallback_with_invalid_level():
    _configure_stdlib_fallback("GARBAGE")


@pytest.mark.skipif(not _STRUCTLOG_AVAILABLE, reason="structlog is installed")
def test_structlog_logger_has_expected_methods():
    logger = get_logger("furrow.test")
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")
    assert hasattr(logger, "error")


@pytest.mark.skipif(_STRUCTLOG_AVAILABLE, reason="structlog is NOT installed")
def test_fallback_logger_is_stdlib_logger():
    logger = get_logger("furrow.test")
    assert isinstance(logger, logging.Logger)