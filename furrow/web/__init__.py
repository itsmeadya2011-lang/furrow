"""Web subpackage.

The FastAPI app and its runner are re-exported lazily (:pep:`562``
``__getattr__``) for the same import-cycle reason as the sibling packages:
keeping the package initialisation side-effect free means importing
``furrow.web`` never drags in ``furrow.llm`` -> ``furrow.core`` ->
``furrow.agents`` and back around again.  Deferring to first attribute access
preserves ``furrow.web.app`` and ``from furrow.web import app`` exactly as
before.  Do not revert to eager imports.
"""

from __future__ import annotations

from typing import Any

__all__ = ["app", "run"]


def __getattr__(name: str) -> Any:
    if name == "app":
        from furrow.web.server import app

        return app
    if name == "run":
        from furrow.web.server import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
