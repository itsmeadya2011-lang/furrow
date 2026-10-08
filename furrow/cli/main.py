from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.config import Provider, settings

console = Console()


@click.group()
@click.option("--provider", type=click.Choice([p.value for p in Provider]), default=None, help="LLM provider")
@click.option("--model", default=None, help="LLM model to use")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="Workspace directory")
@click.option("--state-file", default=None, type=click.Path(), help="State file path")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), default=None, help="Log level")
@click.pass_context
def main(
    ctx: click.Context,
    provider: str | None,
    model: str | None,
    max_cycles: int | None,
    workspace: str | None,
    state_file: str | None,
    log_level: str | None,
) -> None:
    """Furrow - Autonomous coding agent with infinite parallel development loop."""
    ctx.ensure_object(dict)
    
    # Update settings from CLI options
    if provider:
        settings.provider = Provider(provider)
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if workspace:
        settings.workspace = Path(workspace).resolve()
    if state_file:
        settings.state_file = state_file
    if log_level:
        settings.log_level = log_level
    
    ctx.obj["settings"] = settings


@main.command()
@click.argument("goal", required=False)
@click.option("--no-persist", is_flag=True, help="Disable state persistence")
@click.pass_context
def start(ctx: click.Context, goal: str | None, no_persist: bool) -> None:
    """Start the Furrow development loop with a goal."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    settings = ctx.obj["settings"]
    if no_persist:
        settings.persist_state = False
    
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=settings.max_cycles).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    """Start the Furrow web interface."""
    from furrow.web.server import run
    settings.web_host = host
    settings.web_port = port
    run(host=host, port=port)


@main.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show current Furrow status."""
    state_file = Path(settings.state_file)
    if not state_file.exists():
        console.print("[yellow]No active Furrow session found.[/yellow]")
        return
    
    try:
        import json
        with open(state_file) as f:
            state = json.load(f)
        
        from rich.table import Table
        table = Table(title="Furrow Status")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        
        table.add_row("Goal", state.get("goal", "Unknown"))
        table.add_row("Cycles", str(state.get("cycles", 0)))
        table.add_row("Total Tasks", str(len(state.get("all_tasks", []))))
        table.add_row("Completed", str(len(state.get("completed_tasks", []))))
        table.add_row("Failed", str(len(state.get("failed_tasks", []))))
        table.add_row("Started", state.get("started_at", "Unknown"))
        table.add_row("Updated", state.get("updated_at", "Unknown"))
        
        console.print(table)
        
        if state.get("all_tasks"):
            console.print("\n[bold]Tasks:[/bold]")
            task_table = Table(show_header=True, header_style="bold blue")
            task_table.add_column("ID", style="cyan")
            task_table.add_column("Description", style="white")
            task_table.add_column("Status", style="white")
            
            for task in state["all_tasks"]:
                status_style = "green" if task.get("status") == "completed" else "red" if task.get("status") == "failed" else "yellow"
                task_table.add_row(
                    task["id"],
                    task["description"][:60],
                    f"[{status_style}]{task.get('status', 'pending')}[/{status_style}]"
                )
            
            console.print(task_table)
    
    except Exception as e:
        console.print(f"[red]Error reading state: {e}[/red]")


@main.command()
@click.confirmation_option(prompt="Are you sure you want to clear the Furrow state?")
def clear_state() -> None:
    """Clear the Furrow state file."""
    state_file = Path(settings.state_file)
    if state_file.exists():
        state_file.unlink()
        console.print(f"[green]Cleared state file: {state_file}[/green]")
    else:
        console.print("[yellow]No state file to clear.[/yellow]")


if __name__ == "__main__":
    main()