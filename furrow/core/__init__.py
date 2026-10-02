"""Core subpackage.

``Orchestrator`` is re-exported lazily (:pep:`562` ``__getattr__``) because an
eager import here closes an import cycle: ``furrow.llm`` imports
``furrow.core.json_utils``, which forces this package to initialise, and
``furrow.core.orchestrator`` imports ``furrow.llm`` -- leaving ``furrow.llm``
half-initialised.  Deferring the import to first attribute access breaks the
cycle and preserves the public surface.  Do not revert to eager imports.
"""

from __future__ import annotations

from typing import Any

__all__ = ["Orchestrator"]


def __getattr__(name: str) -> Any:
    if name == "Orchestrator":
        from furrow.core.orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
