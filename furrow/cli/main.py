from __future__ import annotations

import asyncio
import os
import sys

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


@click.group()
def main() -> None:
    pass


def _missing_api_key_error() -> str | None:
    """Return a clear error message if the API key for the configured provider is unset."""
    if settings.provider == Provider.ANTHROPIC:
        if not (settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")):
            return (
                "No API key set for provider 'anthropic'. "
                "Set the ANTHROPIC_API_KEY environment variable "
                "(or FURROW_ANTHROPIC_API_KEY) and try again."
            )
    elif settings.provider == Provider.OPENAI:
        if not (settings.openai_api_key or os.getenv("OPENAI_API_KEY")):
            return (
                "No API key set for provider 'openai'. "
                "Set the OPENAI_API_KEY environment variable "
                "(or FURROW_OPENAI_API_KEY) and try again."
            )
    return None


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
def start(goal: str | None, model: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model

    api_key_error = _missing_api_key_error()
    if api_key_error:
        console.print(f"[red]Error:[/red] {api_key_error}")
        sys.exit(1)

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except ValueError as e:
        # Configuration problems (e.g. missing API key raised by the LLM client)
        console.print(f"[red]Configuration error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(
            f"[red]Furrow encountered an unexpected error:[/red] {type(e).__name__}: {e}"
        )
        sys.exit(1)


@main.command()
def web() -> None:
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
