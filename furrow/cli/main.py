from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
@click.option("--workspace", default=None, type=click.Path(exists=True), help="Workspace directory")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default=None, help="Log level")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int,
    max_parallel: int | None,
    workspace: str | None,
    log_level: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Apply overrides
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider) if isinstance(provider, str) else provider
    if max_cycles:
        settings.max_cycles = max_cycles
    if max_parallel:
        settings.max_parallel_tasks = max_parallel
    if workspace:
        settings.workspace = Path(workspace) if isinstance(workspace, str) else workspace
    if log_level:
        settings.log_level = log_level

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
