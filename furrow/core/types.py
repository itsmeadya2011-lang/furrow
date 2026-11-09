"""Shared type definitions for Furrow.

This module provides common type definitions used across the Furrow codebase
to improve type safety and enable better IDE support.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypedDict

from furrow.config import RunResult as ConfigRunResult

if TYPE_CHECKING:
    from furrow.core.orchestrator import Orchestrator


class RunStatus(TypedDict):
    """Status snapshot of a run, returned by Orchestrator.status().

    Attributes:
        goal: The current working goal (may include test failure feedback).
        original_goal: The original user-provided goal.
        cycles: Number of cycles completed so far.
        stopped_reason: Human-readable reason why the run stopped.
        tasks: List of task dictionaries with their current status.
        usage: Accumulated LLM usage statistics.
    """

    goal: str
    original_goal: str
    cycles: int
    stopped_reason: str
    tasks: list[dict[str, Any]]
    usage: dict[str, Any]


class RunRegistryEntry(TypedDict):
    """Entry in the web server's run registry.

    Attributes:
        running: Whether a run is currently in progress.
        orchestrator: The active orchestrator instance, or None.
        result: The result of the completed run, or None.
        task: The asyncio task running the orchestrator, or None.
        error: Error message if the run failed, or None.
    """

    running: bool
    orchestrator: Orchestrator | None
    result: ConfigRunResult | None
    task: asyncio.Task[ConfigRunResult] | None
    error: str | None


# Re-export the actual RunResult from config for convenience
RunResult = ConfigRunResult

__all__ = [
    "RunRegistryEntry",
    "RunResult",
    "RunStatus",
]