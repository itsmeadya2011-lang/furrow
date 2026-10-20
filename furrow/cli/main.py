from __future__ import annotations

import asyncio
import os
import sys

import click
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator
from furrow.logging import configure_logging

console = Console()


def _missing_credential(provider: Provider) -> str | None:
    """Return the name of the missing credential env var, or None if configured."""
    if provider == Provider.ANTHROPIC:
        if settings.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"):
            return None
        return "ANTHROPIC_API_KEY"
    if provider == Provider.OPENAI:
        if settings.openai_api_key or os.getenv("OPENAI_API_KEY"):
            return None
        return "OPENAI_API_KEY"
    return None


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama"]), default=None)
@click.option("--max-cycles", type=int, default=None, help="0 = unlimited")
@click.option("--workspace", type=click.Path(), default=None)
@click.option("--planner-model", default=None)
@click.option("--worker-model", default=None)
@click.option("--tester-model", default=None)
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    workspace: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    if model:
        settings.model = model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if workspace:
        from pathlib import Path

        settings.workspace = Path(workspace)
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model

    missing = _missing_credential(settings.provider)
    if missing is not None:
        console.print(
            f"Configuration error: {missing} is not set. "
            "Set it in your environment, pass --provider openai, or use --provider ollama for a local model."
        )
        sys.exit(1)

    configure_logging(settings.log_level)
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except (ValueError, RuntimeError, click.ClickException) as e:
        msg = str(e)
        if "api_key" in msg.lower() or "apikey" in msg.lower() or "authentication" in msg.lower():
            console.print(f"[red]Configuration error:[/red] {e}")
            sys.exit(1)
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@main.command()
@click.option("--host", default="0.0.0.0")
@click.option("--port", type=int, default=8000)
def web(host: str, port: int) -> None:
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
