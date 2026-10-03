"""Furrow: an autonomous coding agent with an iterative plan/execute/test loop."""

import logging

from furrow.config import Settings, settings
from furrow.llm import LLMClient

__all__ = ["LLMClient", "Settings"]

logger = logging.getLogger(__name__)


def _resolve_level(level: str) -> int:
    """Translate a ``Settings.log_level`` name (or number) into a logging level.

    Falls back to ``INFO`` for anything unrecognized so a typo in
    ``FURROW_LOG_LEVEL`` (or in a ``.env`` file) degrades to a sane default
    instead of raising during import.
    """
    name = level.strip().upper()
    if name.isdigit():
        return int(name)
    named = getattr(logging, name, None)
    return named if isinstance(named, int) else logging.INFO


# Honour Settings.log_level so the ``logging`` calls made across the package
# actually emit output. ``basicConfig`` is a no-op once handlers exist, so an
# embedding application that configured logging first keeps its own setup.
logging.basicConfig(level=_resolve_level(settings.log_level))
