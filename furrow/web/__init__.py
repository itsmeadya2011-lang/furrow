"""Web interface for Furrow.

This package provides a FastAPI-based web server with:
- Single-page UI for starting/monitoring runs
- WebSocket endpoint for real-time event streaming
- REST API for programmatic control
- Persisted state viewing

Run with: `furrow web` or `python -m furrow.web.server`
"""

from furrow.web.server import app, run

__all__ = ["app", "run"]
