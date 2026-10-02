"""The furrow command line entry point.

This package exists so ``furrow.cli`` is a real package: setuptools' ``find`` (not
``find_namespace``) only collects directories with an ``__init__.py``, so without this
file ``furrow.cli.main`` is missing from a built wheel even though
``[project.scripts] furrow = "furrow.cli.main:main"`` depends on it.

``main`` is re-exported lazily through :pep:`562`'s module level ``__getattr__``.
Importing ``furrow.cli`` therefore stays cheap and does not drag in the click group,
the orchestrator or the LLM client, which keeps ``python -m furrow.cli.main`` from
tripping CPython's "found in sys.modules after import of package" RuntimeWarning.
The name is still resolved (and cached) by the first attribute access, so
``from furrow.cli import main`` yields the click group exactly as before.

Note that importing the submodule sets ``furrow.cli.main`` to the *module*;
``__getattr__`` deliberately rebinds the attribute to the click group so the two
spellings keep behaving the same way they did with a plain re-export.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from furrow.cli.main import main

__all__ = ["main"]


def __getattr__(name: str) -> Any:
    """Resolve ``main`` on first use, keeping ``import furrow.cli`` side-effect free.

    Args:
        name: The attribute being looked up on this package.

    Returns:
        The click group exported by :mod:`furrow.cli.main`.

    Raises:
        AttributeError: For any name other than ``main``.
    """
    if name != "main":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from furrow.cli.main import main

    globals()["main"] = main
    return main


def __dir__() -> list[str]:
    """Advertise the lazily exported names to ``dir()`` and tab completion."""
    return sorted([*globals(), *__all__])