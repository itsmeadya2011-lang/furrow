# This file is load-bearing, not decoration: ``pyproject.toml`` selects packages
# with ``setuptools.packages.find(include = ["furrow*"])``, which only reports
# directories that contain an ``__init__.py``. Without it the built wheel ships
# no ``furrow/cli/`` at all, and the declared console script
# (``furrow = "furrow.cli.main:main"``) points at a module that is not installed.
#
# Deliberately empty of re-exports. Importing ``main`` here would bind the name
# ``furrow.cli.main`` on this package to the click ``Group``, shadowing the
# submodule of the same name -- so ``import furrow.cli.main as cli_main`` would
# hand callers the group instead of the module. Import from
# ``furrow.cli.main`` directly.

__all__: list[str] = []
