from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.panel import Panel

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--resume", is_flag=True, help="Resume from saved state file")
def start(goal: str | None, model: str | None, resume: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    try:
        asyncio.run(Orchestrator(goal=goal, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


@main.command()
def status() -> None:
    """Show current state from saved state file."""
    from furrow.config import settings
    import json
    from pathlib import Path
    
    state_file = settings.state_file
    if not state_file.exists():
        console.print("[yellow]No state file found.[/yellow]")
        return
    
    try:
        data = json.loads(state_file.read_text())
        console.print(Panel.fit(f"[bold green]Furrow State[/bold green]", title="State"))
        console.print(f"Goal: {data.get('goal', 'N/A')}")
        console.print(f"Current Cycle: {data.get('current_cycle', 0)}")
        console.print(f"Completed Cycles: {data.get('completed_cycles', 0)}")
        console.print(f"Created: {data.get('created_at', 'N/A')}")
        console.print(f"Updated: {data.get('updated_at', 'N/A')}")
        console.print(f"\nTasks ({len(data.get('tasks', []))}):")
        for task in data.get('tasks', []):
            status_color = {"pending": "yellow", "completed": "green", "failed": "red"}.get(task.get('status'), "white")
            console.print(f"  [{status_color}]{task.get('status').upper()}[/{status_color}] {task.get('id')}: {task.get('description')}")
            if task.get('result'):
                console.print(f"    Result: {task.get('result')[:100]}...")
    except Exception as e:
        console.print(f"[red]Error reading state: {e}[/red]")


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear the saved state?")
def clear_state() -> None:
    """Clear the saved state file."""
    from furrow.config import settings
    state_file = settings.state_file
    if state_file.exists():
        state_file.unlink()
        console.print("[green]State file cleared.[/green]")
    else:
        console.print("[yellow]No state file to clear.[/yellow]")


if __name__ == "__main__":
    main()
