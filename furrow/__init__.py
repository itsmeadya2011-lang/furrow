"""Top-level ``furrow`` package.

Re-exports are resolved lazily via :pep:`562` ``__getattr__`` rather than with
eager ``from ... import ...`` statements.  Eager re-exports create a genuine
import cycle: ``furrow.llm`` -> ``furrow.core.json_utils`` -> ``furrow.core`` ->
``furrow.core.orchestrator`` -> ``furrow.agents.planner`` -> ``furrow.agents`` ->
back to the half-initialised ``furrow.llm``.  Importing this package would then
fail with "cannot import name 'LLMClient' from partially initialized module".
Deferring the submodule import to first attribute access breaks the cycle while
keeping the public surface (``furrow.LLMClient``, ``from furrow import
LLMClient``, ``dir(furrow)``) byte-for-byte identical.  Do not "simplify" this
back into eager imports.
"""

from __future__ import annotations

from typing import Any

__all__ = ["LLMClient", "Settings"]


def __getattr__(name: str) -> Any:
    if name == "LLMClient":
        from furrow.llm import LLMClient

        return LLMClient
    if name == "Settings":
        from furrow.config import Settings

        return Settings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
