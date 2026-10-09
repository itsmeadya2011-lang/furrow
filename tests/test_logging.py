from furrow.logging import configure_logging, get_logger


def test_configure_logging_does_not_raise():
    configure_logging("INFO")
    configure_logging("DEBUG")


def test_get_logger_returns_bound_logger():
    logger = get_logger("test")
    assert logger is not None


def test_get_logger_with_name():
    logger = get_logger("furrow.test")
    assert logger is not None
