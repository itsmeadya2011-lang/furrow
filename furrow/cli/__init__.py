"""Furrow's command line entry point.

Intentionally free of imports: importing :mod:`furrow.cli.main` here would pull
``click``/``rich`` into every ``import furrow.cli`` and can create a circular
import with :mod:`furrow.web.server`.
"""