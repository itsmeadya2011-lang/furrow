from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None, help="LLM provider")
@click.option("--max-cycles", default=None, type=int, help="Maximum number of cycles (0 = unlimited)")
@click.option("--max-parallel", default=None, type=int, help="Maximum parallel tasks")
@click.option("--workspace", default=None, type=click.Path(exists=True, file_okay=False, dir_okay=True), help="Workspace directory")
@click.option("--cycle-timeout", default=None, type=int, help="Timeout per cycle in seconds")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: str | None,
    cycle_timeout: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    config = Settings(
        provider=Provider(provider) if provider else settings.provider,
        model=model or settings.model,
        max_cycles=max_cycles if max_cycles is not None else settings.max_cycles,
        max_parallel_tasks=max_parallel if max_parallel is not None else settings.max_parallel_tasks,
        workspace=Path(workspace) if workspace else settings.workspace,
        cycle_timeout=cycle_timeout if cycle_timeout is not None else settings.cycle_timeout,
    )

    try:
        asyncio.run(Orchestrator(goal=goal, config=config).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()