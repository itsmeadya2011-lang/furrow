from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--workspace", "-w", default=None, type=click.Path(path_type=Path), help="Working directory for Furrow")
@click.option("--cycles", "-c", default=None, type=int, help="Maximum number of development cycles (0 = unlimited)")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
@click.option("--version", "-v", "show_version", is_flag=True, help="Show version and exit")
def start(
    goal: str | None,
    model: str | None,
    workspace: Path | None,
    cycles: int | None,
    provider: str | None,
    show_version: bool,
) -> None:
    if show_version:
        try:
            from importlib.metadata import version
            console.print(f"furrow {version('furrow')}")
        except Exception:
            console.print("furrow 0.1.0")
        sys.exit(0)

    if workspace:
        if not workspace.exists() or not workspace.is_dir():
            console.print(f"[red]Error: workspace directory does not exist: {workspace}[/red]")
            sys.exit(1)
        settings.workspace = workspace.resolve()

    if cycles is not None:
        settings.max_cycles = cycles

    if provider:
        settings.provider = Provider(provider)

    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        settings.model = model

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
