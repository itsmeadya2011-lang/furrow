"""Tests for furrow.logging configuration behavior."""

from __future__ import annotations

import logging

import pytest

import furrow.logging as flogging


@pytest.fixture(autouse=True)
def _restore_root_logging():
    """Snapshot and restore global logging state around each test."""
    root = logging.root
    saved_handlers = root.handlers[:]
    saved_level = root.level
    try:
        yield
    finally:
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


def test_configure_logging_with_structlog_calls_configure(monkeypatch):
    """When structlog is available, configure_logging calls structlog.configure."""
    calls = []

    class _FakeStructlog:
        contextvars = type("CV", (), {"merge_contextvars": object()})
        processors = type(
            "P",
            (),
            {
                "add_log_level": object(),
                "TimeStamper": staticmethod(lambda **kw: object()),
                "StackInfoRenderer": staticmethod(lambda *a, **kw: object()),
                "format_exc_info": object(),
                "UnicodeDecoder": staticmethod(lambda *a, **kw: object()),
                "JSONRenderer": staticmethod(lambda *a, **kw: object()),
            },
        )

        @staticmethod
        def configure(**kwargs):
            calls.append(kwargs)

        @staticmethod
        def make_filtering_bound_logger(level):
            return object()

        stdlib = type("S", (), {"LoggerFactory": staticmethod(lambda: object())})

    monkeypatch.setattr(flogging, "_STRUCTLOG_AVAILABLE", True)
    monkeypatch.setattr(flogging, "structlog", _FakeStructlog)

    # Should not raise.
    flogging.configure_logging("INFO")

    assert len(calls) == 1
    assert "processors" in calls[0]
    assert "logger_factory" in calls[0]


def test_configure_logging_without_structlog_uses_stdlib_format(monkeypatch):
    """Without structlog, the rich stdlib fallback format must actually apply."""
    monkeypatch.setattr(flogging, "_STRUCTLOG_AVAILABLE", False)
    monkeypatch.setattr(flogging, "structlog", None)

    flogging.configure_logging("DEBUG")

    handlers = logging.root.handlers
    assert handlers, "expected a root handler to be configured"

    formats = []
    for handler in handlers:
        fmt = getattr(handler.formatter, "_fmt", None)
        if fmt is not None:
            formats.append(fmt)

    assert any("%(asctime)s" in f and "%(levelname)s" in f for f in formats), formats
    assert logging.root.level == logging.DEBUG


def test_get_logger_returns_stdlib_when_unavailable(monkeypatch):
    """get_logger returns a stdlib logger when structlog is unavailable."""
    monkeypatch.setattr(flogging, "_STRUCTLOG_AVAILABLE", False)
    monkeypatch.setattr(flogging, "structlog", None)

    logger = flogging.get_logger("some.name")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "some.name"


def test_get_logger_returns_structlog_when_available(monkeypatch):
    """get_logger returns a structlog logger when structlog is available."""
    marker = object()

    class _FakeStructlog:
        @staticmethod
        def get_logger(name):
            return marker

    monkeypatch.setattr(flogging, "_STRUCTLOG_AVAILABLE", True)
    monkeypatch.setattr(flogging, "structlog", _FakeStructlog)

    logger = flogging.get_logger("some.name")

    assert logger is marker