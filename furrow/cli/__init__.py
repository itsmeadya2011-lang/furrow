"""Command-line interface for Furrow.

This ``__init__.py`` is load-bearing. ``[project.scripts]`` declares
``furrow = "furrow.cli.main:main"``, so ``furrow.cli`` has to be a real package
for the console script to be importable from a wheel or an editable install --
without it, ``furrow.cli`` is only an implicit namespace package, which every
other subpackage in this project is not.

The click group lives in :mod:`furrow.cli.main` and is re-exported here, so
``from furrow.cli import main`` works as well as ``from furrow.cli.main import
main``. ``python -m furrow`` is served by :mod:`furrow.__main__`.
"""

from furrow.cli.main import main

__all__ = ["main"]
