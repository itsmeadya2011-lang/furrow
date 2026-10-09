from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    """Furrow: Autonomous coding agent with infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--workspace", default=None, type=click.Path(), help="Workspace directory")
def start(goal: str | None, model: str | None, max_cycles: int | None, workspace: str | None) -> None:
    """Start the Furrow development loop with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    settings = Settings.from_workspace(
        workspace=__import__("pathlib").Path(workspace) if workspace else None
    )
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles

    try:
        asyncio.run(Orchestrator(goal=goal, settings=settings).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def web(host: str, port: int) -> None:
    """Start the Furrow web UI server."""
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
@click.argument("file", default="furrow.json")
def init(file: str) -> None:
    """Initialize a furrow.json config file in the current directory."""
    import json
    from pathlib import Path

    config = {
        "$schema": "https://furrow.ai/schema/config.json",
        "name": Path.cwd().name,
        "description": "Autonomous coding agent with infinite parallel development loop",
        "version": "0.1.0",
        "settings": {
            "max_parallel_tasks": 5,
            "max_cycles": 0,
            "workspace": ".",
        },
    }
    Path(file).write_text(json.dumps(config, indent=2))
    console.print(f"[green]Created {file}[/green]")


if __name__ == "__main__":
    main()
