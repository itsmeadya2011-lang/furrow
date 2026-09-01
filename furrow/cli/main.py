from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"]),
    default=None,
    help="LLM provider to use",
)
@click.option("--max-cycles", type=int, default=None, help="Maximum number of cycles")
@click.option("--workspace", type=click.Path(path_type=Path), default=None, help="Workspace directory")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    workspace: Path | None,
) -> None:
    from furrow.config import settings

    if provider is not None:
        settings.provider = provider
    if model is not None:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if workspace is not None:
        settings.workspace = workspace

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def version() -> None:
    console.print("furrow 0.1.0")


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
