"""Tests for :mod:`furrow.logging`.

These tests are deterministic, touch no network or external service, and
restore the stdlib root logger to a clean state so they never leave global
logging configuration broken for other test modules.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from furrow import logging as furrow_logging


def _reset_root_logger() -> None:
    """Return the stdlib root logger to a clean, unconfigured state.

    ``logging.basicConfig`` is a no-op once the root logger already has
    handlers, so tests that assert on the configured level must start from a
    clean root logger to be deterministic regardless of test order.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)


def test_get_logger_returns_logger_like_object() -> None:
    """get_logger returns a usable logger whether or not structlog is installed."""
    logger = furrow_logging.get_logger("furrow.tests.logging")

    assert logger is not None
    assert callable(getattr(logger, "info", None))
    assert callable(getattr(logger, "error", None))

    if furrow_logging._STRUCTLOG_AVAILABLE:
        assert hasattr(logger, "bind")
    else:
        assert isinstance(logger, logging.Logger)


def test_get_logger_falls_back_to_stdlib_when_structlog_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With structlog unavailable, get_logger returns a stdlib Logger."""
    monkeypatch.setattr(furrow_logging, "_STRUCTLOG_AVAILABLE", False)
    monkeypatch.setattr(furrow_logging, "structlog", None)

    logger = furrow_logging.get_logger("furrow.tests.fallback")

    assert isinstance(logger, logging.Logger)
    assert logger.name == "furrow.tests.fallback"


def test_configure_logging_sets_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """configure_logging("DEBUG") configures the stdlib root logger level."""
    monkeypatch.setattr(furrow_logging, "_STRUCTLOG_AVAILABLE", False)
    monkeypatch.setattr(furrow_logging, "structlog", None)

    _reset_root_logger()
    try:
        furrow_logging.configure_logging("DEBUG")

        assert logging.getLogger().level == logging.DEBUG
        assert len(logging.getLogger().handlers) == 1
    finally:
        _reset_root_logger()


def test_configure_logging_unknown_level_defaults_to_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unrecognised level falls back to INFO instead of raising."""
    monkeypatch.setattr(furrow_logging, "_STRUCTLOG_AVAILABLE", False)
    monkeypatch.setattr(furrow_logging, "structlog", None)

    _reset_root_logger()
    try:
        furrow_logging.configure_logging("NOT_A_LEVEL")

        assert logging.getLogger().level == logging.INFO
    finally:
        _reset_root_logger()


def test_configure_logging_structlog_branch_configures_structlog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When structlog is available, configure_logging configures structlog."""
    pytest.importorskip("structlog")

    monkeypatch.setattr(furrow_logging, "_STRUCTLOG_AVAILABLE", True)

    configured: dict[str, Any] = {}

    def fake_configure(**kwargs: Any) -> None:
        configured.update(kwargs)

    monkeypatch.setattr(furrow_logging.structlog, "configure", fake_configure)

    _reset_root_logger()
    try:
        furrow_logging.configure_logging("INFO")
    finally:
        _reset_root_logger()

    assert "processors" in configured
    assert "wrapper_class" in configured
    assert "logger_factory" in configured
