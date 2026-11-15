import logging

from furrow.logging import configure_logging, get_logger


def test_configure_logging_can_be_called_without_error():
    configure_logging()


def test_get_logger_returns_a_logger_object():
    logger = get_logger("furrow.test.logger")
    assert logger is not None


def test_logger_can_log_messages_at_various_levels():
    configure_logging(level="DEBUG")
    logger = get_logger("furrow.test.levels")
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")


def test_logger_respects_configured_level():
    configure_logging(level="WARNING")
    logger = get_logger("furrow.test.respects_level")
    if hasattr(logger, "isEnabledFor"):
        assert logger.isEnabledFor(logging.WARNING) is True
        assert logger.isEnabledFor(logging.DEBUG) is False
