from __future__ import annotations

import asyncio
import json
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from furrow.config import settings
from furrow.core.orchestrator import STATE_FILE, Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", type=int, default=None, help="Maximum number of cycles")
def start(goal: str | None, model: str | None, provider: str | None, max_cycles: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        from furrow.config import Provider
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def status() -> None:
    """Show current Furrow state."""
    if not STATE_FILE.exists():
        console.print("[yellow]No active Furrow session.[/yellow]")
        return

    try:
        with open(STATE_FILE) as f:
            data = json.load(f)

        console.print(Panel.fit("[bold green]Furrow Status[/bold green]", title="Status"))
        console.print(f"Goal: {data.get('goal', 'Unknown')}")
        console.print(f"Cycle: {data.get('cycles', 0)}")

        tasks = data.get("tasks", [])
        if tasks:
            table = Table(title="Tasks")
            table.add_column("ID", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Status", style="green")
            table.add_column("Result", style="dim")

            for t in tasks:
                status_style = {"completed": "green", "failed": "red", "pending": "yellow"}.get(t.get("status"), "white")
                table.add_row(
                    t.get("id", ""),
                    t.get("description", "")[:60],
                    f"[{status_style}]{t.get('status', '')}[/{status_style}]",
                    (t.get("result", "") or "")[:80],
                )
            console.print(table)
        else:
            console.print("No tasks yet.")
    except Exception as e:
        console.print(f"[red]Failed to read state: {e}[/red]")


@main.command()
def stop() -> None:
    """Stop the current Furrow session by clearing state."""
    if STATE_FILE.exists():
        STATE_FILE.unlink()
        console.print("[green]Furrow session stopped. State cleared.[/green]")
    else:
        console.print("[yellow]No active session to stop.[/yellow]")


@main.command()
def config() -> None:
    """Show current configuration."""
    table = Table(title="Furrow Configuration")
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="white")

    for field_name in settings.model_fields:
        value = getattr(settings, field_name)
        if field_name.endswith("_key") and value:
            value = "***"
        table.add_row(field_name, str(value))
    console.print(table)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


if __name__ == "__main__":
    main()
