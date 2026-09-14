from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console
from rich.panel import Panel

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--resume", is_flag=True, help="Resume from previous state file")
@click.option("--workspace", default=None, type=click.Path(exists=True), help="Workspace directory")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"]), help="Log level")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int,
    provider: str | None,
    resume: bool,
    workspace: str | None,
    log_level: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    
    config = Settings()
    if model:
        config.model = model
    if max_cycles > 0:
        config.max_cycles = max_cycles
    if provider:
        config.provider = Provider(provider)
    if workspace:
        config.workspace = workspace
    if log_level:
        config.log_level = log_level
    
    try:
        asyncio.run(Orchestrator(goal=goal, config=config, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=8000, type=int, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run
    run(host=host, port=port)


@main.command()
@click.option("--workspace", default=".", type=click.Path(exists=True), help="Workspace directory")
def status(workspace: str) -> None:
    """Show the current Furrow state."""
    from furrow.config import FurrowState
    from pathlib import Path
    
    state_file = Path(workspace) / ".furrow_state.json"
    if state_file.exists():
        state = FurrowState.load(state_file)
        console.print(Panel.fit(
            f"[bold]Goal:[/bold] {state.goal}\n"
            f"[bold]Current Cycle:[/bold] {state.current_cycle}\n"
            f"[bold]Total Cycles:[/bold] {len(state.cycles)}\n"
            f"[bold]Created:[/bold] {state.created_at}\n"
            f"[bold]Updated:[/bold] {state.updated_at}",
            title="Furrow Status"
        ))
        for cycle in state.cycles:
            console.print(f"\n[bold cyan]Cycle {cycle.cycle}[/bold cyan]")
            for task in cycle.plan.tasks:
                status_color = "green" if task.status == "completed" else "red" if task.status == "failed" else "yellow"
                console.print(f"  [{status_color}]{task.id}[/{status_color}]: {task.description} ({task.status})")
    else:
        console.print("[yellow]No state file found.[/yellow]")


@main.command()
@click.option("--workspace", default=".", type=click.Path(exists=True), help="Workspace directory")
def clean(workspace: str) -> None:
    """Clean the Furrow state file."""
    from pathlib import Path
    
    state_file = Path(workspace) / ".furrow_state.json"
    if state_file.exists():
        state_file.unlink()
        console.print("[green]State file cleaned.[/green]")
    else:
        console.print("[yellow]No state file found.[/yellow]")


if __name__ == "__main__":
    main()
