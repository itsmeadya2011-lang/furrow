"""Web monitoring UI.

Re-exports the two names callers need: the ASGI ``app`` to mount elsewhere and the
:func:`~furrow.web.server.run` blocking entry point used by the CLI's ``furrow web``.
"""

from furrow.web.server import app, run

__all__ = ["app", "run"]
