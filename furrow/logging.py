"""Logging configuration for Furrow.

Configures structlog to wrap stdlib logging when available, with a defensive
fallback to stdlib logging if structlog is missing.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping, MutableMapping
from typing import Any


def _configure_stdlib_fallback(level: str = "INFO") -> None:
    """Configure standard library logging as a fallback."""
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
    )


_SECRET_KEY_PATTERNS = [
    re.compile(r"\b(api[_-]?key|apikey|secret|token|password|passwd|auth)\b", re.IGNORECASE),
]


def _redact_secrets(
    _: Any, __: str, event_dict: MutableMapping[str, Any]
) -> Mapping[str, Any]:
    """Redact sensitive values from log event dictionaries.

    Looks for keys matching common secret patterns and replaces their values
    with "[REDACTED]". Also redacts string values that look like API keys
    (e.g., "sk-...", "Bearer ...").
    """
    redacted = {}
    for key, value in event_dict.items():
        # Redact by key name
        if any(pattern.search(key) for pattern in _SECRET_KEY_PATTERNS):
            redacted[key] = "[REDACTED]"
            continue
        # Redact by value pattern (API keys, Bearer tokens, etc.)
        if isinstance(value, str) and (
            re.match(r"^(sk-(?:ant-|proj-)?|Bearer\s+\S+)$", value.strip())
            or re.match(r"^[A-Za-z0-9_-]{20,}$", value.strip())
        ):
            redacted[key] = "[REDACTED]"
            continue
        redacted[key] = value
    return redacted


try:
    import structlog
    _STRUCTLOG_AVAILABLE = True
except ImportError:
    structlog = None  # type: ignore[assignment]
    _STRUCTLOG_AVAILABLE = False


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog to wrap stdlib logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(message)s",
        level=log_level,
        stream=sys.stdout,
    )

    if not _STRUCTLOG_AVAILABLE:
        # Fall back to stdlib logging if structlog is not available
        _configure_stdlib_fallback(level)
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            _redact_secrets,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    """Return a structlog logger, falling back to stdlib if needed."""
    if _STRUCTLOG_AVAILABLE:
        return structlog.get_logger(name)
    return logging.getLogger(name)
