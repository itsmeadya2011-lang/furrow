from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator
from furrow.config import Settings
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings = Settings(model=model)
    else:
        from furrow.config import settings as global_settings
        settings = Settings(**global_settings.model_dump())
    try:
        client = LLMClient(settings=settings)
        asyncio.run(Orchestrator(goal=goal, client=client).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
