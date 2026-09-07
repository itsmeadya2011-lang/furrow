from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Max cycles before stopping (0 = infinite)")
@click.option("--no-test", is_flag=True, help="Skip testing phase")
@click.option("--workspace", type=click.Path(exists=True), help="Working directory")
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int,
    no_test: bool,
    workspace: str | None,
    verbose: bool,
) -> None:
    from furrow.config import settings, setup_logging

    if verbose:
        setup_logging("DEBUG")
    else:
        setup_logging(settings.log_level)

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles:
        settings.max_cycles = max_cycles
    if workspace:
        from pathlib import Path
        settings.workspace = Path(workspace)

    try:
        orchestrator = Orchestrator(
            goal=goal,
            max_cycles=settings.max_cycles,
            no_test=no_test,
            workspace=settings.workspace,
        )
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=8000, type=int, help="Port number")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
