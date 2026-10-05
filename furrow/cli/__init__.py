"""Command line entry point for furrow.

This module re-exports the Click group so ``from furrow.cli import main`` works
alongside the ``furrow = "furrow.cli.main:main"`` console script.  It must exist
for ``furrow.cli`` to be discovered by ``setuptools.packages.find``; without it
the subpackage is silently dropped from the built wheel and the installed entry
point dies with ``ModuleNotFoundError: No module named 'furrow.cli'``.
"""

from __future__ import annotations

from furrow.cli.main import main

__all__ = ["main"]