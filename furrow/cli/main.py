from __future__ import annotations

import asyncio

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--max-cycles",
    type=click.IntRange(min=0),
    default=None,
    help="Maximum planning cycles; 0 means unlimited",
)
def start(goal: str | None, model: str | None, max_cycles: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        from furrow.config import settings
        settings.model = model
    try:
        result = asyncio.run(Orchestrator(goal=goal, max_cycles=max_cycles).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        raise click.exceptions.Exit(0)
    except Exception as exc:
        console.print(f"[red]Furrow failed: {exc}[/red]")
        raise click.exceptions.Exit(1)

    console.print(f"Run {result.status.value}: {result.message}")
    raise click.exceptions.Exit(result.exit_code)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
