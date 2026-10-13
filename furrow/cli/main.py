from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import setup_logging

console = Console()


@click.group()
@click.pass_context
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option("--log-format", default="json", type=click.Choice(["json", "console"]), help="Log format")
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Workspace directory")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
@click.option("--model", default=None, help="Override LLM model")
def main(ctx: click.Context, log_level: str, log_format: str, workspace: Path | None, max_cycles: int, provider: str | None, model: str | None) -> None:
    """Furrow - Autonomous coding agent with infinite parallel development loop."""
    # Apply settings overrides
    settings = Settings(
        log_level=log_level,
        log_format=log_format,
        workspace=workspace or Path.cwd(),
        max_cycles=max_cycles,
    )
    if provider:
        settings.provider = Provider(provider)
    if model:
        settings.model = model
    
    # Setup logging
    setup_logging(settings)
    ctx.ensure_object(dict)
    ctx.obj["settings"] = settings


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False, path_type=Path), help="Workspace directory")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option("--log-format", default=None, type=click.Choice(["json", "console"]), help="Log format")
@click.pass_context
def start(ctx: click.Context, goal: str | None, model: str | None, max_cycles: int, workspace: Path | None, provider: str | None, log_level: str | None, log_format: str | None) -> None:
    """Start a new Furrow development loop."""
    settings = ctx.obj.get("settings") if ctx.obj else None
    if settings is None:
        settings = Settings()
    if model:
        settings.model = model
    if max_cycles:
        settings.max_cycles = max_cycles
    if workspace:
        settings.workspace = workspace
    if provider:
        settings.provider = Provider(provider)
    if log_level:
        settings.log_level = log_level
    if log_format:
        settings.log_format = log_format
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    """Start the Furrow web server."""
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
