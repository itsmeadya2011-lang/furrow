from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=None, type=int, help="Maximum cycles to run")
@click.option("--resume", is_flag=True, help="Resume from previous state")
@click.option("--summary", is_flag=True, help="Show summary after completion")
def start(goal: str | None, model: str | None, max_cycles: int | None, resume: bool, summary: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    orchestrator = Orchestrator(goal=goal, resume=resume)
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    
    if summary:
        _print_summary(orchestrator)


@main.command()
def status() -> None:
    """Show current furrow state."""
    state_file = Path(settings.workspace) / ".furrow_state.json"
    if not state_file.exists():
        console.print("[yellow]No saved state found.[/yellow]")
        return
    
    try:
        data = json.loads(state_file.read_text())
        console.print(f"[bold]Original Goal:[/bold] {data.get('original_goal', 'N/A')}")
        console.print(f"[bold]Current Goal:[/bold] {data.get('current_goal', 'N/A')}")
        console.print(f"[bold]Cycles:[/bold] {data.get('cycles', 0)}")
        
        tasks = data.get('all_tasks', [])
        if tasks:
            table = Table(title="Tasks", show_header=True)
            table.add_column("ID", style="cyan")
            table.add_column("Description", style="white")
            table.add_column("Status", style="green")
            table.add_column("Files", style="blue")
            for t in tasks:
                status_color = "green" if t.get('status') == 'completed' else "red" if t.get('status') == 'failed' else "yellow"
                table.add_row(
                    t.get('id', ''),
                    t.get('description', '')[:60],
                    f"[{status_color}]{t.get('status', '')}[/{status_color}]",
                    ", ".join(t.get('files', []))[:40]
                )
            console.print(table)
        
        history = data.get('cycle_history', [])
        if history:
            console.print(f"\n[bold]Cycle History:[/bold] {len(history)} cycles")
            for h in history[-5:]:  # Show last 5
                test_status = "✓" if h.get('test_passed') else "✗" if h.get('test_passed') is False else "?"
                console.print(f"  Cycle {h['cycle']}: {test_status} {h.get('test_summary', '')[:80]}")
                
    except Exception as e:
        console.print(f"[red]Error reading state: {e}[/red]")


@main.command()
@click.confirmation_option(prompt="Delete saved state?")
def clean() -> None:
    """Remove saved furrow state."""
    state_file = Path(settings.workspace) / ".furrow_state.json"
    if state_file.exists():
        state_file.unlink()
        console.print("[green]State file removed.[/green]")
    else:
        console.print("[yellow]No state file to remove.[/yellow]")


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


def _print_summary(orchestrator: Orchestrator) -> None:
    console.print("\n[bold cyan]═══ Summary ═══[/bold cyan]")
    console.print(f"Goal: {orchestrator.original_goal}")
    console.print(f"Cycles run: {orchestrator.cycles}")
    
    completed = sum(1 for t in orchestrator.all_tasks if t.status == "completed")
    failed = sum(1 for t in orchestrator.all_tasks if t.status == "failed")
    pending = sum(1 for t in orchestrator.all_tasks if t.status == "pending")
    
    console.print(f"Tasks: {completed} completed, {failed} failed, {pending} pending")
    
    for task in orchestrator.all_tasks:
        status_color = "green" if task.status == "completed" else "red" if task.status == "failed" else "yellow"
        console.print(f"  [{status_color}]{task.id}[/{status_color}]: {task.description} ({task.status})")
        if task.result:
            console.print(f"    Result: {task.result[:200]}...")


if __name__ == "__main__":
    main()
