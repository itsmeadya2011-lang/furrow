"""Top-level package exports.

``Orchestrator`` is exposed lazily via :pep:`562` module-level ``__getattr__`` so
that importing :mod:`furrow` does not pull in the core orchestration machinery
(and its transitive dependencies) unless it is actually used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from furrow.config import Settings
from furrow.llm import LLMClient

if TYPE_CHECKING:
    from furrow.core.orchestrator import Orchestrator

__all__ = ["LLMClient", "Orchestrator", "Settings"]


def __getattr__(name: str) -> Any:
    if name == "Orchestrator":
        from furrow.core.orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals(), *__all__})