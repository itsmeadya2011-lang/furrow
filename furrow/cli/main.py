from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
@click.option("--workspace", default=None, help="Workspace directory")
@click.option("--log-level", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
@click.pass_context
def main(ctx: click.Context, workspace: str | None, log_level: str) -> None:
    ctx.ensure_object(dict)
    if workspace:
        settings.workspace = Path(workspace).resolve()
    settings.log_level = log_level.upper()
    import logging
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--resume", is_flag=True, help="Resume from previous session state")
@click.option("--state-file", default=None, help="Path to state file for persistence")
@click.pass_context
def start(ctx: click.Context, goal: str | None, model: str | None, provider: str | None, max_cycles: int | None, resume: bool, state_file: str | None) -> None:
    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")
    
    if model:
        settings.model = model
    if provider:
        settings.provider = provider
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if state_file:
        settings.state_file = Path(state_file)
    
    # Validate API keys
    client = LLMClient()
    try:
        if settings.provider.value == "anthropic" and not (settings.anthropic_api_key or ctx.obj.get("ANTHROPIC_API_KEY")):
            console.print("[red]Error: ANTHROPIC_API_KEY not set[/red]")
            sys.exit(1)
        elif settings.provider.value == "openai" and not (settings.openai_api_key or ctx.obj.get("OPENAI_API_KEY")):
            console.print("[red]Error: OPENAI_API_KEY not set[/red]")
            sys.exit(1)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)
    
    try:
        asyncio.run(Orchestrator(goal=goal or "", client=client, settings=settings, resume=resume).run())
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
@click.option("--state-file", default=None, help="Path to state file")
def status(state_file: str | None) -> None:
    """Show the current state of the Furrow session."""
    from furrow.config import StateManager
    if state_file:
        state_path = Path(state_file)
    else:
        state_path = settings.state_file
    
    state = StateManager(state_path)
    
    if not state.current_goal:
        console.print("[yellow]No active session found.[/yellow]")
        return
    
    console.print(Panel.fit(f"[bold green]Furrow Session[/bold green]\nGoal: {state.current_goal}", title="Status"))
    
    table = Table(title="Task Status")
    table.add_column("ID", style="cyan")
    table.add_column("Description", style="white")
    table.add_column("Status", style="bold")
    table.add_column("Cycle", style="dim")
    table.add_column("Attempts", style="dim")
    
    for task in state.all_tasks:
        status_style = {
            "completed": "[green]✓[/green]",
            "failed": "[red]✗[/red]",
            "in_progress": "[yellow]⟳[/yellow]",
            "pending": "[dim]○[/dim]",
        }.get(task.status, task.status)
        table.add_row(task.id, task.description[:50], status_style, str(task.cycle), str(task.attempts))
    
    console.print(table)
    console.print(f"Cycles completed: {len(state.cycles)}")


@main.command()
@click.option("--state-file", default=None, help="Path to state file")
def reset(state_file: str | None) -> None:
    """Reset the Furrow session state."""
    if state_file:
        state_path = Path(state_file)
    else:
        state_path = settings.state_file
    
    if state_path.exists():
        state_path.unlink()
        console.print("[green]State reset.[/green]")
    else:
        console.print("[yellow]No state file found.[/yellow]")


if __name__ == "__main__":
    main()
