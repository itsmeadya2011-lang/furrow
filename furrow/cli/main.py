from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator, FEEDBACK_FILE
from furrow.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)

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
        from furrow.config import settings
        settings.model = model
    logger.info("furrow_started", goal=goal, model=model)
    console.print(f"[dim]Tip: Write to {FEEDBACK_FILE} to provide mid-loop feedback[/dim]")
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        logger.info("furrow_stopped_by_user")
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.argument("feedback", required=False)
def feedback(feedback: str | None) -> None:
    """Provide feedback to a running Furrow session."""
    if not feedback:
        feedback = click.prompt("Enter your feedback")
    feedback_path = Path.cwd() / FEEDBACK_FILE
    feedback_path.write_text(feedback)
    console.print(f"[green]Feedback written to {feedback_path}[/green]")


@main.command()
def web() -> None:
    logger.info("furrow_web_started")
    from furrow.web.server import run
    run()


if __name__ == "__main__":
    main()
