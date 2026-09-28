from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from furrow.config import Provider, configure_logging, settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider], case_sensitive=False),
    default=None,
    help="LLM provider (anthropic, openai, ollama)",
)
@click.option(
    "--workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Working directory for the agent",
)
@click.option(
    "--max-cycles",
    type=click.IntRange(min=0),
    default=None,
    help="Maximum number of planning cycles (0 = unlimited)",
)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    workspace: Path | None,
    max_cycles: int | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    configure_logging()
    overrides: dict[str, Any] = {}
    if model:
        overrides["model"] = model
    if provider:
        overrides["provider"] = Provider(provider)
    if workspace:
        overrides["workspace"] = workspace
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    local_settings = settings.model_copy(update=overrides)
    client = LLMClient(settings=local_settings)
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
