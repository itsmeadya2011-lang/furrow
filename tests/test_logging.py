"""Tests for furrow.logging."""

from __future__ import annotations

import logging

from furrow.logging import _configure_stdlib_fallback, configure_logging, get_logger


def test_configure_logging_sets_stdlib_level(monkeypatch):
    """configure_logging should set the stdlib root logger level."""
    seen: dict[str, int] = {}

    def fake_basicConfig(*, format, level, stream):
        seen["level"] = level

    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)
    monkeypatch.setattr("furrow.logging._STRUCTLOG_AVAILABLE", False)

    configure_logging("DEBUG")
    assert seen["level"] == logging.DEBUG


def test_configure_logging_default_level(monkeypatch):
    """configure_logging defaults to INFO when given an invalid level name."""
    seen: dict[str, int] = {}

    def fake_basicConfig(*, format, level, stream):
        seen["level"] = level

    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)
    monkeypatch.setattr("furrow.logging._STRUCTLOG_AVAILABLE", False)

    configure_logging("not-a-level")
    assert seen["level"] == logging.INFO


def test_configure_stdlib_fallback_writes_to_stdout(monkeypatch, capsys):
    """_configure_stdlib_fallback should configure stdlib logging to stdout."""
    captured: dict[str, object] = {}

    def fake_basicConfig(*, format, level, stream):
        captured["stream"] = stream
        captured["level"] = level

    monkeypatch.setattr(logging, "basicConfig", fake_basicConfig)
    _configure_stdlib_fallback("WARNING")
    assert captured["level"] == logging.WARNING
    assert captured["stream"] is not None


def test_get_logger_returns_logger():
    """get_logger should return a logger object (structlog or stdlib)."""
    logger = get_logger("furrow.test")
    assert logger is not None
    # Either a structlog bound logger or a stdlib Logger.
    assert hasattr(logger, "info") or hasattr(logger, "debug")


def test_get_logger_different_names(monkeypatch):
    """get_logger with different names returns different loggers."""
    monkeypatch.setattr("furrow.logging._STRUCTLOG_AVAILABLE", False)
    a = get_logger("furrow.a")
    b = get_logger("furrow.b")
    assert a is not b
    assert a.name == "furrow.a"
    assert b.name == "furrow.b"
