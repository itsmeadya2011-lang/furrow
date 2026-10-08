from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.panel import Panel

from furrow.config import Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", type=int, default=None, help="Maximum number of cycles")
@click.option("--max-parallel", type=int, default=None, help="Maximum parallel tasks")
@click.option("--workspace", type=click.Path(exists=True, file_okay=False), default=None, help="Workspace directory")
@click.option("--resume/--no-resume", default=True, help="Resume from saved state")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: str | None,
    resume: bool,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    # Update settings
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if workspace:
        settings.workspace = workspace
    
    client = LLMClient(settings=settings)
    
    try:
        asyncio.run(Orchestrator(goal=goal, client=client, settings=settings).run())
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
@click.option("--workspace", type=click.Path(exists=True, file_okay=False), default=".", help="Workspace directory")
def status(workspace: str) -> None:
    """Show current Furrow state."""
    from pathlib import Path
    import json
    
    state_path = Path(workspace) / ".furrow_state.json"
    if not state_path.exists():
        console.print("[yellow]No active Furrow state found.[/yellow]")
        return
    
    try:
        with open(state_path) as f:
            data = json.load(f)
        
        console.print(Panel.fit(f"[bold green]Furrow State[/bold green]", title="Status"))
        console.print(f"Goal: {data.get('goal', 'N/A')}")
        console.print(f"Cycles: {data.get('cycles', 0)}")
        
        tasks = data.get('tasks', [])
        completed = data.get('completed_tasks', [])
        all_tasks = tasks + completed
        
        if all_tasks:
            from rich.table import Table
            table = Table(title="Tasks")
            table.add_column("ID")
            table.add_column("Description")
            table.add_column("Status")
            table.add_column("Files")
            for task in all_tasks:
                table.add_row(
                    task.get('id', ''),
                    task.get('description', '')[:50],
                    task.get('status', ''),
                    ", ".join(task.get('files', [])[:3])
                )
            console.print(table)
        else:
            console.print("No tasks yet.")
    except Exception as e:
        console.print(f"[red]Error reading state: {e}[/red]")


@main.command()
@click.option("--workspace", type=click.Path(exists=True, file_okay=False), default=".", help="Workspace directory")
def clean(workspace: str) -> None:
    """Clean up Furrow state file."""
    from pathlib import Path
    
    state_path = Path(workspace) / ".furrow_state.json"
    if state_path.exists():
        state_path.unlink()
        console.print("[green]State file removed.[/green]")
    else:
        console.print("[yellow]No state file to remove.[/yellow]")


if __name__ == "__main__":
    main()