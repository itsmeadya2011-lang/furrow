"""Shared runner utilities for CLI and web server.

This module contains common functionality for running the orchestrator
and managing LLM client lifecycle, eliminating duplication between
the CLI and web server entry points.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any, Protocol

from furrow.config import Settings
from furrow.config import RunResult as RunResultType
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import StateStore

logger = logging.getLogger(__name__)


class ClosableClient(Protocol):
    """Protocol for an LLM client that can be closed."""
    def aclose(self) -> Any | Awaitable[Any]: ...


async def release_client(client: ClosableClient | None) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may not exist yet and may be a plain or an async
    method, so it is looked up defensively and only awaited when the result
    is awaitable.  A failing close is logged and ignored: closing must never
    turn a successful run into a failed one.
    """
    if client is None:
        return
    closer = getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as exc:
        logger.warning("Could not close the LLM client: %s", exc)


async def run_orchestrator(orchestrator: Orchestrator) -> RunResultType:
    """Run an orchestrator and release its LLM client afterwards.

    The client is closed in a ``finally`` block, so its HTTP connections are
    released when the run finishes, when it fails and when it is cancelled.
    """
    try:
        return await orchestrator.run()
    finally:
        await release_client(getattr(orchestrator, "client", None))


def create_orchestrator(
    goal: str,
    settings: Settings,
    resume: bool = False,
    state_store: StateStore | None = None,
) -> Orchestrator:
    """Create an orchestrator with the given settings and optional state store."""
    if state_store is None:
        state_store = StateStore(settings.state_path())
    return Orchestrator(
        goal=goal,
        settings=settings,
        state_store=state_store,
        resume=resume,
    )


def create_state_store(settings: Settings) -> StateStore:
    """Create a state store for the given settings."""
    return StateStore(settings.state_path())


def build_run_settings(
    base_settings: Settings,
    *,
    model: str | None = None,
    max_cycles: int | None = None,
    workspace: Path | str | None = None,
) -> Settings:
    """Build a validated Settings instance with the given overrides applied.

    The base settings object is never mutated: a fresh validated instance is
    constructed for this run only.
    """
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if workspace is not None:
        updates["workspace"] = Path(workspace)
    merged = {**base_settings.model_dump(), **updates}
    return Settings.model_validate(merged)