from __future__ import annotations

import asyncio
import os
import sys

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator, configure_logging
from furrow.llm import LLMClient

console = Console()

MAX_GOAL_LENGTH = 5000


def validate_goal(goal: str) -> str:
    if not goal or not goal.strip():
        raise click.BadParameter("Goal cannot be empty")
    if len(goal) > MAX_GOAL_LENGTH:
        raise click.BadParameter(f"Goal exceeds maximum length of {MAX_GOAL_LENGTH} characters")
    return goal.strip()


def check_api_keys() -> None:
    provider = settings.provider.value
    if provider == "anthropic":
        if not settings.anthropic_api_key and not os.getenv("ANTHROPIC_API_KEY"):
            raise click.ClickException(
                "ANTHROPIC_API_KEY is not set. Please set it via FURROW_ANTHROPIC_API_KEY environment variable, "
                "in a .env file, or use --provider openai with OPENAI_API_KEY."
            )
    elif provider == "openai":
        if not settings.openai_api_key and not os.getenv("OPENAI_API_KEY"):
            raise click.ClickException(
                "OPENAI_API_KEY is not set. Please set it via FURROW_OPENAI_API_KEY environment variable, "
                "in a .env file, or use --provider anthropic with ANTHROPIC_API_KEY."
            )


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--resume", is_flag=True, default=False, help="Resume from saved state")
@click.option("--verbose", is_flag=True, default=False, help="Enable debug-level logging")
def start(goal: str | None, model: str | None, resume: bool, verbose: bool) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    try:
        goal = validate_goal(goal)
    except click.BadParameter as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    if model:
        settings.model = model

    configure_logging(verbose=verbose)

    try:
        check_api_keys()
    except click.ClickException as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)

    try:
        asyncio.run(Orchestrator(goal=goal).run(resume=resume))
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except RuntimeError as e:
        if "API key" in str(e) or "authentication" in str(e).lower():
            console.print(f"[red]Authentication error:[/red] {e}")
            console.print("[dim]Please check your API key is valid and has sufficient quota.[/dim]")
        else:
            console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option("--verbose", is_flag=True, default=False, help="Enable debug-level logging")
def web(verbose: bool) -> None:
    configure_logging(verbose=verbose)
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
