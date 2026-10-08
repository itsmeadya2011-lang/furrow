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
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--state-file", default=None, help="Path to state file for persistence")
@click.option("--workspace", default=None, help="Workspace directory")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int,
    state_file: str | None,
    workspace: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    if workspace:
        settings.workspace = Path(workspace).resolve()

    state_path = Path(state_file).resolve() if state_file else None

    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_path).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
@click.option("--state-file", default=".furrow_state.json", help="State file to inspect")
def status(state_file: str) -> None:
    """Show current state from state file."""
    import json
    path = Path(state_file)
    if not path.exists():
        console.print(f"[red]State file not found: {path}[/red]")
        return

    data = json.loads(path.read_text())
    console.print(f"[bold]Goal:[/bold] {data.get('goal', 'N/A')}")
    console.print(f"[bold]Cycles:[/bold] {data.get('cycles', 0)}")
    tasks = data.get('tasks', [])
    console.print(f"[bold]Tasks:[/bold] {len(tasks)}")
    for task in tasks:
        status_color = {"completed": "green", "failed": "red", "pending": "yellow"}.get(task.get('status'), "white")
        console.print(f"  [{status_color}]{task.get('id')}[/]: {task.get('description')} ({task.get('status')})")
        if task.get('result'):
            console.print(f"    Result: {task.get('result')[:100]}")


if __name__ == "__main__":
    main()