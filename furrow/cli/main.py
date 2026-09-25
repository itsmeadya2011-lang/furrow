from __future__ import annotations

import asyncio
import sys

import click
import structlog
from rich.console import Console
from rich.panel import Panel

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

# Configure structlog
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING level by default
)

console = Console()


def setup_logging(level: str) -> None:
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.make_filtering_bound_logger(getattr(__import__('logging'), level.upper(), 30)),
    )


@click.group()
@click.option("--log-level", default="WARNING", help="Log level (DEBUG, INFO, WARNING, ERROR)")
def main(log_level: str) -> None:
    setup_logging(log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice([p.value for p in Provider]), default=None, help="Override LLM provider")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
@click.option("--resume", is_flag=True, help="Resume from saved state")
@click.option("--dry-run", is_flag=True, help="Show plan without executing")
def start(goal: str | None, model: str | None, provider: str | None, max_cycles: int, resume: bool, dry_run: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles > 0:
        console.print(f"[cyan]Max cycles set to: {max_cycles}[/cyan]")
    
    if dry_run:
        console.print(f"[yellow]DRY RUN - Goal: {goal}[/yellow]")
        console.print("Would start Furrow with the above settings.")
        return
    
    try:
        asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles, resume=resume).run())
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
def status() -> None:
    """Show current Furrow state"""
    from furrow.core.orchestrator import STATE_FILE
    if STATE_FILE.exists():
        import json
        data = json.loads(STATE_FILE.read_text())
        console.print(Panel.fit("[bold green]Furrow State[/bold green]", title="Status"))
        console.print(f"Goal: {data.get('goal', 'N/A')}")
        console.print(f"Cycles: {data.get('cycles', 0)}")
        console.print(f"Tasks: {len(data.get('tasks', []))}")
        for task in data.get('tasks', []):
            console.print(f"  - {task['id']}: {task['description']} ({task['status']})")
    else:
        console.print("[yellow]No saved state found.[/yellow]")


if __name__ == "__main__":
    main()
