from __future__ import annotations

import click

from furrow.web.server import run


@click.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web_cmd(host: str, port: int) -> None:
    run(host=host, port=port)