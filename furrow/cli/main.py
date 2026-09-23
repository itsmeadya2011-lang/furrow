from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

console = Console()


@click.group()
@click.option("--log-level", default="INFO", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
@click.option("--json-logs", is_flag=True, help="Output logs as JSON")
def main(log_level: str, json_logs: bool) -> None:
    configure_logging(log_level=log_level, json_output=json_logs)
    # Update settings
    settings.log_level = log_level


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
@click.option("--max-retries", default=None, type=int, help="Maximum retries per task")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    max_retries: int | None
) -> None:
    """Start Furrow with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    if model:
        settings.model = model
    if provider:
        from furrow.config import Provider
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if max_retries is not None:
        settings.max_retries = max_retries

    console.print(f"[bold]Configuration:[/bold]")
    console.print(f"  Provider: {settings.provider.value}")
    console.print(f"  Model: {settings.model}")
    console.print(f"  Max cycles: {settings.max_cycles or 'unlimited'}")
    console.print(f"  Max parallel: {settings.max_parallel_tasks}")
    console.print(f"  Max retries: {settings.max_retries}")
    console.print()

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    """Start the Furrow web UI."""
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
def version() -> None:
    """Show Furrow version."""
    from furrow import __version__
    console.print(f"Furrow {__version__}")


if __name__ == "__main__":
    main()