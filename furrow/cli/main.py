from __future__ import annotations

import asyncio
import sys
from importlib.metadata import version

import click
from rich.console import Console
from rich.panel import Panel

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.version_option(version("furrow"), "-v", "--version", message="Furrow %(version)s")
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--resume", is_flag=True, help="Resume from saved state")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles")
def start(goal: str | None, model: str | None, provider: str | None, resume: bool, max_cycles: int | None) -> None:
    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")

    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    if max_cycles is not None:
        settings.max_cycles = max_cycles

    try:
        asyncio.run(Orchestrator(goal=goal or "", resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind")
@click.option("--port", default=8000, help="Port to bind")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
def status() -> None:
    """Show status of saved state."""
    import json

    from furrow.core.orchestrator import STATE_FILE

    if not STATE_FILE.exists():
        console.print("[yellow]No saved state found.[/yellow]")
        return

    try:
        data = json.loads(STATE_FILE.read_text())
        console.print(Panel.fit("[bold green]Furrow State[/bold green]", title="Status"))
        console.print(f"Goal: {data.get('goal', 'N/A')}")
        console.print(f"Cycles completed: {data.get('cycles', 0)}")
        console.print(f"Tasks: {len(data.get('tasks', []))}")
        console.print(f"History entries: {len(data.get('history', []))}")
        console.print(f"Last updated: {data.get('last_updated', 'N/A')}")
        console.print(f"Final: {data.get('final', False)}")
    except Exception as e:
        console.print(f"[red]Error reading state: {e}[/red]")


@main.command()
def clear_state() -> None:
    """Clear saved state."""
    from furrow.core.orchestrator import STATE_FILE
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        console.print("[green]State cleared.[/green]")
    else:
        console.print("[yellow]No state file to clear.[/yellow]")


if __name__ == "__main__":
    main()