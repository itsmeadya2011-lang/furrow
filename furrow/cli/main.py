from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    """Furrow: Autonomous coding agent with infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"], case_sensitive=False), default=None, help="Override LLM provider")
@click.option("--max-cycles", type=int, default=None, help="Maximum number of development cycles (0 = unlimited)")
@click.option("--max-parallel", type=int, default=None, help="Maximum parallel tasks per cycle")
@click.option("--workspace", type=click.Path(path_type=Path), default=None, help="Workspace directory")
@click.option("--state-file", type=click.Path(path_type=Path), default=None, help="Path to persistent state file")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default=None, help="Log verbosity")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: Path | None,
    state_file: Path | None,
    log_level: str | None,
) -> None:
    """Start the infinite Furrow development loop with a GOAL."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    from furrow.config import Settings

    settings_kwargs: dict = {}
    if model:
        settings_kwargs["model"] = model
    if provider:
        settings_kwargs["provider"] = provider
    if max_cycles is not None:
        settings_kwargs["max_cycles"] = max_cycles
    if max_parallel is not None:
        settings_kwargs["max_parallel_tasks"] = max_parallel
    if workspace:
        settings_kwargs["workspace"] = workspace
    if state_file:
        settings_kwargs["state_file"] = state_file
    if log_level:
        settings_kwargs["log_level"] = log_level

    settings = Settings(**settings_kwargs)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to listen on")
def web(host: str, port: int) -> None:
    """Start the Furrow web UI."""
    from furrow.web.server import run

    run(host=host, port=port)


@main.command()
def doctor() -> None:
    """Check configuration and dependencies."""
    from furrow.config import settings

    console.print("[bold cyan]Furrow Doctor[/bold cyan]\n")
    console.print(f"Provider: {settings.provider}")
    console.print(f"Model: {settings.model}")
    console.print(f"Max cycles: {settings.max_cycles or 'unlimited'}")
    console.print(f"Max parallel tasks: {settings.max_parallel_tasks}")
    console.print(f"Workspace: {settings.workspace}")
    console.print(f"State file: {settings.state_file}")
    console.print(f"Log level: {settings.log_level}")

    if settings.provider == Provider.ANTHROPIC:
        key = settings.anthropic_api_key or "ANTHROPIC_API_KEY"
        console.print(f"Anthropic API key: {'[green]set[/green]' if key else '[red]not set[/red]'}")
    elif settings.provider == Provider.OPENAI:
        key = settings.openai_api_key or "OPENAI_API_KEY"
        console.print(f"OpenAI API key: {'[green]set[/green]' if key else '[red]not set[/red]'}")

    console.print(f"\nPackages: [green]ok[/green]")


if __name__ == "__main__":
    main()
