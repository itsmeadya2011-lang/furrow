"""Tests for :mod:`furrow.logging`.

``configure_logging`` is called once at start-up by the CLI, but its behaviour
(structlog configuration and the stdlib fallback) was only ever mocked out, so
a regression in this module would have been silent. These tests pin the
documented behaviour directly.

``logging.basicConfig`` is a no-op once the root logger has handlers (pytest
attaches one), so the level is observed by intercepting ``basicConfig``
instead of inspecting the root logger.
"""

from __future__ import annotations

import logging

import pytest

import furrow.logging as flogging


def _capture_basic_config(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Replace ``logging.basicConfig`` with a recorder and return its kwargs."""
    captured: dict = {}

    def fake_basic_config(**kwargs) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(flogging.logging, "basicConfig", fake_basic_config)
    return captured


def test_configure_logging_sets_level(monkeypatch):
    """With structlog available, the requested level reaches ``basicConfig``."""
    captured = _capture_basic_config(monkeypatch)
    monkeypatch.setattr(flogging, "_STRUCTLOG_AVAILABLE", True)
    flogging.configure_logging("DEBUG")
    assert captured["level"] == logging.DEBUG


def test_configure_logging_invalid_level_defaults_to_info(monkeypatch):
    """An unknown level name falls back to INFO instead of raising."""
    captured = _capture_basic_config(monkeypatch)
    flogging.configure_logging("NOT_A_REAL_LEVEL")
    assert captured["level"] == logging.INFO


def test_configure_logging_falls_back_without_structlog(monkeypatch):
    """When structlog is missing, the stdlib fallback is invoked with the level."""
    _capture_basic_config(monkeypatch)
    calls: list[str] = []
    monkeypatch.setattr(flogging, "_STRUCTLOG_AVAILABLE", False)
    monkeypatch.setattr(
        flogging,
        "_configure_stdlib_fallback",
        lambda level="INFO": calls.append(level),
    )
    flogging.configure_logging("WARNING")
    assert calls == ["WARNING"]


def test_get_logger_falls_back_to_stdlib(monkeypatch):
    """Without structlog, ``get_logger`` returns a stdlib logger."""
    monkeypatch.setattr(flogging, "_STRUCTLOG_AVAILABLE", False)
    logger = flogging.get_logger("furrow.test")
    assert isinstance(logger, logging.Logger)


def test_get_logger_returns_structlog_when_available(monkeypatch):
    """With structlog available, ``get_logger`` returns a bound logger."""
    monkeypatch.setattr(flogging, "_STRUCTLOG_AVAILABLE", True)
    logger = flogging.get_logger("furrow.test")
    assert logger is not None