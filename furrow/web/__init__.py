"""Web layer for furrow: the FastAPI app plus the blocking ``run`` entry point."""

from __future__ import annotations

from furrow.web.server import app, run

__all__ = ["app", "run"]