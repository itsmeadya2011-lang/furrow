"""CLI package for Furrow.

Re-exports the Click ``main`` entry point so it can be imported as
``from furrow.cli import main`` as well as ``from furrow.cli.main import main``.
"""

from furrow.cli.main import main

__all__ = ["main"]
