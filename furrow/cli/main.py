from __future__ import annotations

import asyncio
import os
import sys
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient
from pathlib import Path

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--workspace", default=None, type=click.Path(), help="Working directory for Furrow")
@click.option("--max-cycles", default=0, type=int, help="Maximum cycles to run (0 = unlimited)")
@click.option("--provider", default=None, type=click.Choice(["anthropic", "openai", "ollama"]), help="LLM provider")
def start(goal: str | None, model: str | None, workspace: str | None, max_cycles: int, provider: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if provider:
        overrides["provider"] = Provider(provider)
    if max_cycles:
        overrides["max_cycles"] = max_cycles
    if workspace:
        overrides["workspace"] = Path(workspace)

    if overrides:
        local_settings = settings.model_copy(update=overrides)
        client = LLMClient(settings=local_settings)
    else:
        client = None

    if workspace:
        os.chdir(workspace)

    try:
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
