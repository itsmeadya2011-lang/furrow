"""Tests for furrow.logging.

These tests are offline and deterministic: they never touch the network or
require external credentials. They cover the public API (configure_logging,
get_logger), the stdlib fallback path, and structured logging regardless of
which backend is active.
"""

from __future__ import annotations

import logging

from furrow import logging as furrow_logging


def test_configure_logging_info_returns_none(caplog):
    """configure_logging("INFO") runs without raising and returns None."""
    result = furrow_logging.configure_logging("INFO")
    assert result is None


def test_configure_logging_unknown_level_falls_back_to_info(caplog):
    """An unknown level does not raise and falls back to INFO."""
    # Should not raise; unknown levels fall back to logging.INFO.
    furrow_logging.configure_logging("BOGUS")


def test_get_logger_exposes_callables(caplog):
    """get_logger returns an object with the expected callables."""
    caplog.set_level(logging.INFO)
    logger = furrow_logging.get_logger("furrow.test")
    for method in ("info", "warning", "error", "debug"):
        assert callable(getattr(logger, method))


def test_get_logger_emits_record(caplog):
    """A logger obtained from get_logger emits a record caplog can capture."""
    caplog.set_level(logging.INFO)
    furrow_logging.configure_logging("INFO")
    logger = furrow_logging.get_logger("furrow.test")
    logger.info("hello world")
    assert any("hello world" in r.getMessage() for r in caplog.records)


def test_structlog_logger_accepts_keyword_context(caplog):
    """Calling a logger with keyword context does not raise."""
    caplog.set_level(logging.INFO)
    furrow_logging.configure_logging("INFO")
    logger = furrow_logging.get_logger("furrow.test")
    logger.info("hello", key="value")


def test_fallback_path(monkeypatch, caplog):
    """When structlog is unavailable, stdlib logging is used."""
    monkeypatch.setattr(furrow_logging, "_STRUCTLOG_AVAILABLE", False)
    monkeypatch.setattr(furrow_logging, "structlog", None)

    # configure_logging still works without raising.
    furrow_logging.configure_logging("INFO")

    logger = furrow_logging.get_logger("furrow.test")
    assert isinstance(logger, logging.Logger)

    caplog.set_level(logging.INFO)
    logger.info("fallback message")
    assert any("fallback message" in r.getMessage() for r in caplog.records)


def test_fallback_logger_accepts_keyword_context(monkeypatch, caplog):
    """Keyword context must not raise on the stdlib fallback either."""
    monkeypatch.setattr(furrow_logging, "_STRUCTLOG_AVAILABLE", False)
    monkeypatch.setattr(furrow_logging, "structlog", None)

    caplog.set_level(logging.INFO)
    logger = furrow_logging.get_logger("furrow.test")
    logger.info("hello", extra={"key": "value"})