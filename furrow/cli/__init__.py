"""CLI subpackage.

The click group ``main`` is re-exported lazily (:pep:`562` ``__getattr__``) for
the same import-cycle reason as the sibling packages: ``furrow.cli.main`` imports
``furrow.core.orchestrator`` at module level, which pulls in ``furrow.agents``,
``furrow.llm`` and ``furrow.config``.  An eager import here would make merely
importing ``furrow.cli`` (which the ``furrow`` console script entry point does)
initialise that entire graph, so importing ``furrow.cli.main`` from anywhere
low in it re-enters a half-initialised module.  Deferring to first attribute
access breaks the cycle and preserves the public surface
(``from furrow.cli import main``).  Do not revert to eager imports.

Note also that this file must exist: ``[tool.setuptools.packages.find]`` only
discovers directories that are importable packages, i.e. those carrying an
``__init__.py``.  Without it ``furrow.cli`` is dropped from the wheel and the
``furrow = "furrow.cli.main:main"`` console script fails with ``ImportError``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["main"]


def __getattr__(name: str) -> Any:
    if name == "main":
        from .main import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
