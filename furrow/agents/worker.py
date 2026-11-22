"""Worker agent for the Furrow autonomous coding system.

This module re-exports the worker agent and its utilities from the worker package
for backward compatibility.
"""

from __future__ import annotations

from furrow.agents.worker.agent import WorkerAgent
from furrow.agents.worker.command_policy import (
    MAX_LIST_ENTRIES,
    MAX_READ_CHARS,
    TRUNCATED_STOP_REASONS,
    _extract_command_substitutions,
    is_command_denied,
    resolve_in_workspace,
)

__all__ = [
    "WorkerAgent",
    "WorkerResult",
    "MAX_LIST_ENTRIES",
    "MAX_READ_CHARS",
    "TRUNCATED_STOP_REASONS",
    "is_command_denied",
    "resolve_in_workspace",
    "_extract_command_substitutions",
]

# Re-export WorkerResult from config for backward compatibility
from furrow.config import WorkerResult