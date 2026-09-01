from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    """Furrow - Autonomous coding agent with an infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model for all agents")
@click.option("--cycles", default=None, type=int, help="Maximum number of cycles to run (0 = unlimited)")
@click.option("--workspace", default=None, type=click.Path(exists=True), help="Workspace directory")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
def start(
    goal: str | None,
    model: str | None,
    cycles: int | None,
    workspace: str | None,
    provider: str | None,
) -> None:
    from furrow.config import settings

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        settings.model = model
    if cycles is not None:
        settings.max_cycles = cycles
    if workspace:
        settings.workspace = workspace
    if provider:
        settings.provider = provider

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind to")
@click.option("--port", default=8000, show_default=True, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
