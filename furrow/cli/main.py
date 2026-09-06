from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--state", default=None, help="Path to state file for persistence")
@click.option("--resume", is_flag=True, help="Resume from existing state file")
def start(goal: str | None, model: str | None, state: str | None, resume: bool) -> None:
    from pathlib import Path

    if not goal and not resume:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        runtime_settings = Settings(model=model)
    else:
        from furrow.config import settings as runtime_settings
    state_path = Path(state) if state else None
    try:
        asyncio.run(
            Orchestrator(
                goal=goal or "",
                settings=runtime_settings,
                state_path=state_path,
                resume=resume,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
