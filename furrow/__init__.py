from furrow.config import Settings
from furrow.llm import LLMClient

__version__ = "0.1.0"

__all__ = ["LLMClient", "Settings", "__version__", "configure_logging"]


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for the Furrow package.

    Call this at application startup (CLI, web server) to set up
    structured logging with the given level.
    """
    import logging

    import structlog

    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if level.upper() in ("DEBUG", "INFO") else structlog.processors.JSONRenderer(),
        ],
    )
