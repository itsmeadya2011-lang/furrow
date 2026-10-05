# `main` is deliberately not re-exported here: a package attribute shadows the
# submodule of the same name, so `from furrow.cli import main` would hand back the
# click group instead of the `furrow.cli.main` module that callers and the test
# suite reach for.  `[project.scripts]` resolves `furrow.cli.main:main` through the
# module, so the console script is unaffected either way.
from furrow.cli.main import run_goal

__all__ = ["run_goal"]
