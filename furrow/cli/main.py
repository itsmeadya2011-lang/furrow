from __future__ import annotations

import asyncio
import sys

import click
import structlog
from importlib.metadata import PackageNotFoundError, version
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

try:
    __version__ = version("furrow")
except PackageNotFoundError:
    __version__ = "unknown"

console = Console()
logger = structlog.get_logger(__name__)


@click.group()
@click.version_option(version=__version__)
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        logger.info("Furrow stopped by user.")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
