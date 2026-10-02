from __future__ import annotations

import asyncio
import sys

import click
from rich.console import Console

from furrow.config import settings as default_settings
from furrow.config import settings_for_model
from furrow.core.orchestrator import Orchestrator
from furrow.logging_config import configure_logging, get_logger

console = Console()
log = get_logger(__name__)


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--model",
    default=None,
    help=(
        "Override the LLM model for this run. Applies to the planner, worker and "
        "tester alike, so per-role models (FURROW_PLANNER_MODEL and friends) are "
        "ignored while this flag is set."
    ),
)
def start(goal: str | None, model: str | None) -> None:
    configure_logging(default_settings.log_level)
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    run_settings = settings_for_model(model)
    log.info(
        "run starting",
        goal=goal,
        # What every agent will actually call: settings.model is the value all
        # three agents resolve to once --model is given.
        model=run_settings.model if run_settings is not None else default_settings.model,
        model_override=model is not None,
    )
    try:
        asyncio.run(Orchestrator(goal=goal, settings=run_settings).run())
    except KeyboardInterrupt:
        log.info("run stopped by user")
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    log.info("run finished")


@main.command()
def web() -> None:
    configure_logging(default_settings.log_level)
    log.info("web console starting")
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
