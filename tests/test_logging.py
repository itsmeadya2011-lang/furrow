"""Tests for the Furrow logging configuration."""

from __future__ import annotations

import logging
import sys

import pytest

from furrow import logging as furrow_logging


def test_fallback_configures_stdlib_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without structlog, stdlib logging is configured exactly once.

    Previously configure_logging() called basicConfig unconditionally and then
    the fallback called it a second time with a different format.
    """
    monkeypatch.setattr(furrow_logging, "_STRUCTLOG_AVAILABLE", False)

    calls: list[dict[str, object]] = []

    def fake_basic_config(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(furrow_logging.logging, "basicConfig", fake_basic_config)

    furrow_logging.configure_logging("DEBUG")

    assert len(calls) == 1
    call = calls[0]
    assert call["level"] == logging.DEBUG
    assert call["stream"] is sys.stdout
    assert "%(asctime)s" in str(call["format"])
    assert "%(levelname)s" in str(call["format"])