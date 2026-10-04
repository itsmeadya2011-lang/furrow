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
    """Furrow — autonomous coding agent with an infinite parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the default LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"], case_sensitive=False),
    default=None,
    help="LLM provider",
)
@click.option("--max-cycles", type=int, default=None, help="Max development cycles (0 = unlimited)")
@click.option("--max-parallel", type=int, default=None, help="Max parallel worker tasks")
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Workspace directory",
)
@click.option("--test-command", default=None, help="Explicit test command to run each cycle")
@click.option(
    "--log-level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default=None,
    help="Log level",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: Path | None,
    test_command: str | None,
    log_level: str | None,
) -> None:
    """Start an autonomous development run."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider.lower())
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if workspace is not None:
        settings.workspace = workspace
    if test_command is not None:
        settings.test_command = test_command
    if log_level is not None:
        settings.log_level = log_level
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
def web(host: str, port: int) -> None:
    """Run the web dashboard."""
    from furrow.web.server import run

    run(host=host, port=port)


@main.command()
def version() -> None:
    """Print the Furrow version."""
    click.echo("furrow 0.1.0")


if __name__ == "__main__":
    main()
