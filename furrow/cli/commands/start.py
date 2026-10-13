from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--workspace", default=".", help="Workspace directory")
@click.option("--resume", is_flag=True, help="Resume from previous state")
@click.option("--max-cycles", default=0, help="Maximum number of cycles (0 = unlimited)")
def start_cmd(goal: str | None, model: str | None, workspace: str, resume: bool, max_cycles: int) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if max_cycles > 0:
        settings.max_cycles = max_cycles

    try:
        asyncio.run(Orchestrator(goal=goal, workspace=workspace, resume=resume).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)