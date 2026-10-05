"""Exceptions raised by Furrow agents."""

from __future__ import annotations

__all__ = ["AgentError"]


class AgentError(RuntimeError):
    """Raised when an agent cannot complete its role.

    Subclasses :class:`RuntimeError` so callers that only care about "the agent
    layer failed" can catch a single builtin type.
    """