from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Settings, Provider
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model (e.g., claude-3-5-sonnet-20241022)")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--workspace", default=None, help="Workspace directory")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int,
    workspace: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    config = Settings()
    if model:
        config.model = model
        config.planner_model = model
        config.worker_model = model
        config.tester_model = model
    if provider:
        config.provider = Provider(provider)
    if max_cycles > 0:
        config.max_cycles = max_cycles
    if workspace:
        config.workspace = Path(workspace)

    try:
        asyncio.run(Orchestrator(goal=goal, config=config).run())
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
