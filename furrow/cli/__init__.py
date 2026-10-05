"""Command-line entry point package.

This package previously had no ``__init__.py`` and was only an implicit PEP 420
namespace package. Runtime imports happened to work from a source checkout, but
``setuptools.packages.find()`` skips namespace packages, so built wheels shipped
no CLI module and the declared ``furrow = "furrow.cli.main:main"`` entry point
was broken for every pip-installed user. Adding this module makes the package
discoverable and lets it be imported as ``furrow.cli``.
"""

from furrow.cli.main import main

__all__ = ["main"]
