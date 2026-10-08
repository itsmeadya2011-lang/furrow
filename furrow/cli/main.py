from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.version_option(package_name="furrow")
def main() -> None:
    """Furrow: autonomous coding agent with an infinite parallel development loop."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice([p.value for p in Provider]), default=None, help="LLM provider")
@click.option("--cycles", type=int, default=None, help="Maximum cycles (0 = unlimited)")
@click.option("--max-parallel", type=int, default=None, help="Maximum parallel tasks")
def start(goal: str | None, model: str | None, provider: str | None, cycles: int | None, max_parallel: int | None) -> None:
    """Start the Furrow development loop with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if cycles is not None:
        settings.max_cycles = cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Bind host")
@click.option("--port", default=8000, type=int, help="Bind port")
def web(host: str, port: int) -> None:
    """Run the Furrow web UI."""
    from furrow.web.server import run

    run(host=host, port=port)


@main.command()
def info() -> None:
    """Show Furrow configuration."""
    console.print(f"Provider: {settings.provider.value}")
    console.print(f"Model: {settings.model}")
    console.print(f"Planner model: {settings.planner_model}")
    console.print(f"Worker model: {settings.worker_model}")
    console.print(f"Tester model: {settings.tester_model}")
    console.print(f"Max parallel tasks: {settings.max_parallel_tasks}")
    console.print(f"Max cycles: {settings.max_cycles or 'unlimited'}")


if __name__ == "__main__":
    main()