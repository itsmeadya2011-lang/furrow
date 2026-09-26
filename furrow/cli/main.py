from __future__ import annotations

import asyncio
import logging
import sys

import click
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

console = Console()
log = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
@click.pass_context
def main(ctx: click.Context, log_level: str) -> None:
    _configure_logging(log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    settings = Settings()
    if model:
        settings.model = model
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, help="Bind port")
@click.option("--log-level", default="info", help="Uvicorn log level")
def web(host: str, port: int, log_level: str) -> None:
    from furrow.web.server import run

    run(host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
