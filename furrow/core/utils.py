"""Shared utilities for Furrow CLI and web server.

This module provides common functionality used by both the CLI and the web server
to avoid code duplication. It includes utilities for running orchestrators and
releasing LLM client resources.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)


async def release_client(client: Any) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may not exist yet and may be a plain or an async
    method, so it is looked up defensively and only awaited when the result
    is awaitable. A failing close is logged and ignored: closing must never
    turn a successful run into a failed one.

    Args:
        client: The LLM client to release. Can be any object with an ``aclose``
            or ``close`` method.
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


async def run_orchestrator(orchestrator: Any) -> Any:
    """Run an orchestrator and release its LLM client afterwards.

    The client is closed in a ``finally`` block, so its HTTP connections are
    released when the run finishes, when it fails and when it is cancelled.

    Args:
        orchestrator: The orchestrator instance to run. Must have a ``run``
            method and a ``client`` attribute.

    Returns:
        The result of the orchestrator's ``run`` method.

    Raises:
        Any exception raised by the orchestrator's ``run`` method is propagated
        after the client has been released.
    """
    try:
        return await orchestrator.run()
    finally:
        await release_client(getattr(orchestrator, "client", None))