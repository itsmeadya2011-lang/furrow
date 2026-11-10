"""Shared utilities for Furrow.

This module contains common utility functions used across the Furrow codebase.
"""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from furrow.llm import LLMClient

logger = logging.getLogger(__name__)


async def release_llm_client(client: LLMClient | Any) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may not exist yet and may be a plain or an async
    method, so it is looked up defensively and only awaited when the result
    is awaitable. A failing close is logged and ignored: closing must never
    turn a successful run into a failed one.

    Args:
        client: The LLM client to release. Can be any object with an optional
            ``aclose`` method.
    """
    closer = getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as exc:
        logger.warning("Could not close the LLM client: %s", exc)