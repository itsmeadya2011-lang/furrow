from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
@click.pass_context
def main(ctx: click.Context) -> None:
    ctx.obj = {}


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--max-cycles", default=0, help="Max cycles before auto-halt (0 = unlimited)")
def start(goal: str | None, model: str | None, max_cycles: int) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    settings.max_cycles = max_cycles
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


@main.command()
def doctor() -> None:
    """Run diagnostics on the Furrow environment."""
    from furrow.config import settings
    checks = [
        ("Provider", settings.provider.value),
        ("Model", settings.model),
        ("Planner model", settings.planner_model),
        ("Worker model", settings.worker_model),
        ("Tester model", settings.tester_model),
        ("Max parallel tasks", str(settings.max_parallel_tasks)),
        ("Max cycles", str(settings.max_cycles)),
        ("Workspace", str(settings.workspace)),
    ]
    for name, value in checks:
        console.print(f"[bold]{name}[/bold]: {value}")
    console.print("\n[bold green]All Furrow components are configured.[/bold green]")


@main.command()
def run() -> None:
    """Start the Furrow development loop (default command)."""
    goal = click.prompt("Enter your goal for Furrow")
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
