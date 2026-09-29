"""Tests for the logging module."""
from __future__ import annotations

import logging

import structlog

from furrow.logging import setup_logging


def test_setup_logging_debug() -> None:
    setup_logging("DEBUG")
    logger = structlog.get_logger()
    assert logger is not None
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_info() -> None:
    setup_logging("INFO")
    logger = structlog.get_logger()
    assert logger is not None
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_invalid_level_falls_back_to_info() -> None:
    setup_logging("NOTALEVEL")
    assert logging.getLogger().level == logging.INFO


def test_setup_logging_default_is_info() -> None:
    setup_logging()
    assert logging.getLogger().level == logging.INFO
