# ``main`` is bound to the *module*, not to the click group it defines: exporting
# the group here would shadow ``furrow.cli.main`` on this package, so
# ``import furrow.cli.main as cli`` would hand back a ``click.Group`` instead of
# the module. Every sibling package re-exports its public surface; for this one
# that surface is the module the ``furrow`` console script points at.
from __future__ import annotations

from typing import Any


def __getattr__(name: str) -> Any:
    if name == "main":
        from . import main as _main
        return _main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main"]
