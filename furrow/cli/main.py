from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()

try:
    import structlog

    logger = structlog.get_logger(__name__)
except ImportError:  # pragma: no cover
    logger = None


@click.group()
def main() -> None:
    """Furrow: Autonomous coding agent with an infinite parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    """Start an autonomous development loop for the given goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings

        settings.model = model
    if logger:
        logger.info("cli_start", goal=goal, model=model)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        if logger:
            logger.info("cli_interrupted")
        sys.exit(0)


@main.command()
def web() -> None:
    """Start the Furrow web UI server."""
    if logger:
        logger.info("cli_web_start")
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
