from __future__ import annotations

import structlog

__all__ = ["configure_logging", "get_logger"]


def configure_logging(level: str = "INFO") -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
    )


def get_logger(name: str = "") -> structlog.Logger:
    return structlog.get_logger(name)
