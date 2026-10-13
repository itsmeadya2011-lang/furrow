from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.option("--workspace", default=None, help="Workspace directory")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
def main(workspace: str | None, max_cycles: int | None, model: str | None, provider: str | None, log_level: str | None) -> None:
    if workspace:
        settings.workspace = Path(workspace)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if model:
        settings.model = model
    if provider:
        from furrow.config import Provider
        settings.provider = Provider(provider)
    if log_level:
        settings.log_level = log_level


@main.command()
@click.argument("goal", required=False)
def start(goal: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    try:
        asyncio.run(Orchestrator(goal=goal, workspace=settings.workspace, max_cycles=settings.max_cycles).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
