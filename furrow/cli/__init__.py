"""Command-line interface package for Furrow.

This file exists so that ``furrow.cli`` is discovered by
``setuptools.packages.find`` (``include = ["furrow*"]``) and shipped in
built wheels/sdists. Without it the ``furrow = "furrow.cli.main:main"``
console-script entry point would break for installed users.
"""

from __future__ import annotations