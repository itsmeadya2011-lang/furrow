"""Furrow command line interface."""

from __future__ import annotations

__all__ = ["main"]


def __getattr__(name: str) -> object:
    """Resolve the console entry point lazily so importing ``furrow.cli`` stays cheap."""
    if name == "main":
        from furrow.cli.main import main

        globals()["main"] = main
        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
