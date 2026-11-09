"""Shared utilities for Furrow CLI and web server.

Consolidates code that was duplicated between furrow/cli/main.py and
furrow/web/server.py.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any

from furrow.config import RunResult, Settings
from furrow.core.orchestrator import Orchestrator

logger = logging.getLogger(__name__)


async def _release_client(client: Any) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may not exist yet and may be a plain or an async
    method, so it is looked up defensively and only awaited when the result
    is awaitable.  A failing close is logged and ignored: closing must never
    turn a successful run into a failed one.
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


async def run_orchestrator(orchestrator: Orchestrator) -> RunResult:
    """Run an orchestrator and release its LLM client afterwards.

    The client is closed in a ``finally`` block, so its HTTP connections are
    released when the run finishes, when it fails and when it is cancelled
    (e.g. because the websocket client went away).
    """
    try:
        return await orchestrator.run()
    finally:
        await _release_client(getattr(orchestrator, "client", None))


def build_run_settings(
    settings: Settings,
    model: str | None = None,
    max_cycles: int | None = None,
    workspace: str | None = None,
) -> Settings:
    """Return a validated copy of the given settings with overrides applied.

    The module level ``settings`` object is never mutated: a fresh validated
    instance is constructed for this run only.
    """
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if workspace is not None:
        updates["workspace"] = workspace
    merged = {**settings.model_dump(), **updates}
    return Settings.model_validate(merged)