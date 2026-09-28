import logging

import structlog

from furrow.config import Settings
from furrow.llm import LLMClient

__all__ = ["LLMClient", "Settings"]


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog with a console-friendly renderer."""
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


configure_logging()
