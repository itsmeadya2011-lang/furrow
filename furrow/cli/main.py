from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from furrow.config import Settings, TaskModel
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--resume", is_flag=True, help="Resume from saved state")
def start(goal: str | None, model: str | None, max_cycles: int, resume: bool) -> None:
    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")
    
    settings = Settings()
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    
    state_file = settings.state_file
    
    if resume and not state_file.exists():
        console.print("[red]No saved state found to resume from.[/red]")
        sys.exit(1)
    
    try:
        asyncio.run(Orchestrator(
            goal=goal or "Resume previous session",
            settings=settings,
            state_file=state_file if resume else None,
        ).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def status() -> None:
    """Show current session status."""
    settings = Settings()
    state_file = settings.state_file
    
    if not state_file.exists():
        console.print("[yellow]No active session found.[/yellow]")
        return
    
    import json
    with open(state_file) as f:
        state = json.load(f)
    
    console.print(Panel.fit(f"[bold green]Furrow Session Status[/bold green]", title="Status"))
    console.print(f"Goal: {state.get('goal', 'Unknown')}")
    console.print(f"Cycles: {state.get('cycles', 0)}")
    console.print(f"Last updated: {state.get('timestamp', 'Unknown')}")
    
    tasks = state.get("tasks", [])
    if tasks:
        table = Table(title="Tasks")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Result", style="dim")
        
        for task in tasks:
            status_style = {
                "completed": "green",
                "failed": "red",
                "pending": "yellow",
            }.get(task.get("status", "pending"), "white")
            table.add_row(
                task.get("id", ""),
                task.get("description", "")[:50],
                f"[{status_style}]{task.get('status', 'pending')}[/{status_style}]",
                (task.get("result", "") or "")[:60],
            )
        console.print(table)
    else:
        console.print("No tasks recorded.")


@main.command()
@click.argument("task_id", required=False)
def history(task_id: str | None) -> None:
    """Show task history or details for a specific task."""
    settings = Settings()
    state_file = settings.state_file
    
    if not state_file.exists():
        console.print("[yellow]No session history found.[/yellow]")
        return
    
    import json
    with open(state_file) as f:
        state = json.load(f)
    
    tasks = state.get("tasks", [])
    
    if task_id:
        task = next((t for t in tasks if t.get("id") == task_id), None)
        if task:
            console.print(Panel.fit(f"[bold]Task {task_id}[/bold]", title="Task Details"))
            console.print(f"Description: {task.get('description')}")
            console.print(f"Status: {task.get('status')}")
            console.print(f"Files: {', '.join(task.get('files', []))}")
            console.print(f"Dependencies: {', '.join(task.get('dependencies', []))}")
            console.print(f"Result: {task.get('result', 'No result')}")
        else:
            console.print(f"[red]Task {task_id} not found.[/red]")
    else:
        table = Table(title="Task History")
        table.add_column("ID", style="cyan")
        table.add_column("Description", style="white")
        table.add_column("Status", style="bold")
        table.add_column("Files", style="dim")
        
        for task in tasks:
            status_style = {
                "completed": "green",
                "failed": "red",
                "pending": "yellow",
            }.get(task.get("status", "pending"), "white")
            table.add_row(
                task.get("id", ""),
                task.get("description", "")[:50],
                f"[{status_style}]{task.get('status', 'pending')}[/{status_style}]",
                ", ".join(task.get("files", []))[:40],
            )
        console.print(table)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
