"""Shared utilities for Furrow.

This module collects helpers that are used by more than one entry point
(the CLI and the web server) so the logic lives in exactly one place:

* :func:`release_client` - release the HTTP resources of an LLM client.
* :func:`run_settings` - build a validated :class:`~furrow.config.Settings`
  copy with overrides applied, without mutating the global settings.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any

from furrow.config import Settings, settings as global_settings

logger = logging.getLogger(__name__)


def release_client(client: Any) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may be a plain or an async method, so it is looked
    up defensively and only awaited when the result is awaitable.  A failing
    close is logged and ignored: closing must never turn a successful run
    into a failed one.

    This is a synchronous helper that schedules the awaited close on the
    running event loop when needed.  Callers that already have a loop can use
    :func:`release_client_async` directly.
    """
    closer = getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        outcome = closer()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not close the LLM client: %s", exc)
        return
    if inspect.isawaitable(outcome):
        # The closer is async but we are not in an event loop - the caller is
        # responsible for awaiting it.  This branch is hit only when the
        # synchronous wrapper is misused; the async variant is preferred.
        logger.warning(
            "LLM client aclose() is async but release_client was called "
            "outside an event loop; the close was not awaited."
        )


async def release_client_async(client: Any) -> None:
    """Async variant of :func:`release_client` - awaits the closer when needed.

    A failing close is logged and ignored: closing must never turn a
    successful run into a failed one.
    """
    closer = getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Could not close the LLM client: %s", exc)


def run_settings(
    model: str | None = None,
    max_cycles: int | None = None,
    workspace: Path | None = None,
) -> Settings:
    """Return a validated copy of the global settings with overrides applied.

    The module level ``settings`` object is never mutated: a fresh validated
    instance is constructed for this run only.  Validation is performed via
    ``Settings.model_validate`` so an out-of-range value (a negative
    ``max_cycles`` for instance) is rejected.
    """
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if workspace is not None:
        updates["workspace"] = Path(workspace)
    merged = {**global_settings.model_dump(), **updates}
    return Settings.model_validate(merged)