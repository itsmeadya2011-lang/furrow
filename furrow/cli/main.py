from __future__ import annotations

import asyncio
import sys

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
def start(goal: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    import os
    prev_model = os.environ.get("FURROW_MODEL")
    try:
        if model:
            os.environ["FURROW_MODEL"] = model
        try:
            asyncio.run(Orchestrator(goal=goal).run())
        except KeyboardInterrupt:
            console.print("\n[yellow]Furrow stopped by user.[/yellow]")
            sys.exit(0)
    finally:
        if model:
            if prev_model is None:
                os.environ.pop("FURROW_MODEL", None)
            else:
                os.environ["FURROW_MODEL"] = prev_model


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
