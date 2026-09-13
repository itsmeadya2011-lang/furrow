from __future__ import annotations

import asyncio
import os
import sys

import click
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


def _ensure_credentials(settings: Settings) -> None:
    if settings.provider == Provider.ANTHROPIC:
        if not settings.get_anthropic_key() and not os.getenv("ANTHROPIC_API_KEY"):
            raise click.UsageError(
                "ANTHROPIC_API_KEY is not set. Set it in your environment or .env file."
            )
    elif settings.provider == Provider.OPENAI:
        if not settings.get_openai_key() and not os.getenv("OPENAI_API_KEY"):
            raise click.UsageError(
                "OPENAI_API_KEY is not set. Set it in your environment or .env file."
            )


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
        isolated_settings = Settings(model=model)
    else:
        isolated_settings = Settings()
    _ensure_credentials(isolated_settings)

    async def runner() -> None:
        client = LLMClient(settings=isolated_settings)
        try:
            await Orchestrator(
                goal=goal,
                client=client,
                settings_obj=isolated_settings,
            ).run()
        finally:
            await client.close()

    try:
        asyncio.run(runner())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
