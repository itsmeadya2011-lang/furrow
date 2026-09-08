from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
import structlog
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()
logger = structlog.get_logger(__name__)


@click.group()
def main() -> None:
    """Furrow - Autonomous coding agent with infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = infinite)")
@click.option("--workspace", default=None, type=click.Path(path_type=Path), help="Workspace directory")
def start(goal: str | None, model: str | None, max_cycles: int | None, workspace: Path | None) -> None:
    """Start the Furrow development loop with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    from furrow.config import Settings

    settings = Settings()
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if workspace is not None:
        settings.workspace = workspace

    logger.info("Starting Furrow CLI", goal=goal, model=settings.model, max_cycles=settings.max_cycles)

    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        logger.info("Stopped by user")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    """Start the Furrow web UI server."""
    from furrow.web.server import run

    logger.info("Starting Furrow web UI", host=host, port=port)
    run(host=host, port=port)


if __name__ == "__main__":
    main()
