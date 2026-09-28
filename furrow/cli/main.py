from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import Provider, Settings
from furrow.core.orchestrator import Orchestrator
from furrow.llm import LLMClient

console = Console()


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"], case_sensitive=False), default=None, help="Override LLM provider")
@click.option("--max-cycles", type=int, default=None, help="Maximum number of development cycles (0 = unlimited)")
def start(goal: str | None, model: str | None, provider: str | None, max_cycles: int | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    # Create a fresh Settings instance, optionally overridden from CLI
    settings = Settings()
    if provider:
        settings.provider = Provider(provider)
    if model:
        settings.model = model

    client = LLMClient(settings=settings)

    try:
        asyncio.run(
            Orchestrator(
                goal=goal,
                client=client,
                max_cycles=max_cycles,
            ).run()
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    finally:
        asyncio.run(client.close())


@main.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", type=int, default=8000, help="Port to bind to")
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
