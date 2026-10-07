from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from furrow.config import settings, TaskModel
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    """Furrow - Autonomous coding agent with infinite parallel development loop."""
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--state-file", default=None, help="Path to state file for persistence")
def start(goal: str | None, model: str | None, max_cycles: int, state_file: str | None) -> None:
    """Start Furrow with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles
    try:
        asyncio.run(Orchestrator(goal=goal, state_file=state_file).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    """Start the Furrow web UI."""
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
@click.option("--state-file", default=None, help="Path to state file")
def status(state_file: str | None) -> None:
    """Show current Furrow state from state file."""
    state_path = Path(state_file) if state_file else settings.workspace / ".furrow_state.json"
    if not state_path.exists():
        console.print("[yellow]No state file found.[/yellow]")
        return
    
    try:
        data = json.loads(state_path.read_text())
        console.print(Panel.fit(f"[bold]Furrow State[/bold]", title="Status"))
        
        console.print(f"Goal: {data.get('goal', 'N/A')}")
        console.print(f"Cycles: {data.get('cycles', 0)}")
        console.print(f"Updated: {data.get('updated', 'N/A')}")
        
        tasks = data.get("tasks", [])
        if tasks:
            table = Table(title=f"Tasks ({len(tasks)})")
            table.add_column("ID", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Files", style="dim")
            table.add_column("Status", style="bold")
            table.add_column("Dependencies", style="dim")
            
            for task in tasks:
                status_color = {
                    "pending": "yellow",
                    "completed": "green",
                    "failed": "red",
                }.get(task.get("status", "pending"), "white")
                status_str = f"[{status_color}]{task.get('status', 'pending')}[/{status_color}]"
                table.add_row(
                    task.get("id", ""),
                    task.get("description", "")[:60],
                    ", ".join(task.get("files", [])) if task.get("files") else "any",
                    status_str,
                    ", ".join(task.get("dependencies", [])) if task.get("dependencies") else "none",
                )
            console.print(table)
        else:
            console.print("[dim]No tasks recorded.[/dim]")
    except Exception as e:
        console.print(f"[red]Failed to read state: {e}[/red]")


@main.command()
def config() -> None:
    """Show current configuration."""
    console.print(Panel.fit("[bold]Furrow Configuration[/bold]", title="Config"))
    
    config_data = {
        "provider": settings.provider.value,
        "model": settings.model,
        "planner_model": settings.planner_model,
        "worker_model": settings.worker_model,
        "tester_model": settings.tester_model,
        "max_parallel_tasks": settings.max_parallel_tasks,
        "max_cycles": settings.max_cycles,
        "workspace": str(settings.workspace),
        "log_level": settings.log_level,
        "anthropic_api_key": "***" if settings.anthropic_api_key else "not set",
        "openai_api_key": "***" if settings.openai_api_key else "not set",
        "ollama_base_url": settings.ollama_base_url,
    }
    
    for key, value in config_data.items():
        console.print(f"  [cyan]{key}:[/cyan] {value}")


@main.command()
@click.argument("state_file", required=False)
def resume(state_file: str | None) -> None:
    """Resume from a previous state file."""
    state_path = Path(state_file) if state_file else settings.workspace / ".furrow_state.json"
    if not state_path.exists():
        console.print("[red]State file not found.[/red]")
        sys.exit(1)
    
    try:
        data = json.loads(state_path.read_text())
        goal = data.get("goal", "")
        if not goal:
            console.print("[red]No goal found in state file.[/red]")
            sys.exit(1)
        
        console.print(f"[green]Resuming goal:[/green] {goal}")
        asyncio.run(Orchestrator(goal=goal, state_file=state_file).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Failed to resume: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--state-file", default=None, help="Path to state file")
def clear(state_file: str | None) -> None:
    """Clear the state file."""
    state_path = Path(state_file) if state_file else settings.workspace / ".furrow_state.json"
    if state_path.exists():
        state_path.unlink()
        console.print(f"[green]Cleared state file: {state_path}[/green]")
    else:
        console.print("[yellow]No state file to clear.[/yellow]")


if __name__ == "__main__":
    main()
