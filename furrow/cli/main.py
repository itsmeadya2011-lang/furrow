from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import setup_logging

console = Console()


@click.group()
@click.option("--log-level", default=None, help="Log level (DEBUG, INFO, WARNING, ERROR)")
@click.option("--log-file", default=None, type=click.Path(path_type=Path), help="Log file path")
def main(log_level: str | None, log_file: Path | None) -> None:
    setup_logging(log_level, log_file)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Workspace directory")
@click.option("--resume", is_flag=True, help="Resume from previous state file")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
def start(
    goal: str | None, 
    model: str | None, 
    max_cycles: int,
    workspace: Path | None,
    resume: bool,
    provider: str | None
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    if workspace:
        settings.workspace = workspace
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles, workspace=workspace, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
