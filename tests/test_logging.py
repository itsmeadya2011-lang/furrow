from __future__ import annotations

import logging

from furrow import logging as furrow_logging


def test_configure_logging_stdlib_fallback(monkeypatch) -> None:
    monkeypatch.setattr(furrow_logging, "_STRUCTLOG_AVAILABLE", False)
    furrow_logging.configure_logging("DEBUG")
    logger = furrow_logging.get_logger("furrow.test")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "furrow.test"


def test_get_logger_returns_logger() -> None:
    logger = furrow_logging.get_logger("furrow.test.get")
    assert logger is not None
