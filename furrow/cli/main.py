from __future__ import annotations

import asyncio
import logging
import sys

import click
import structlog
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()
log = structlog.get_logger("furrow.cli")


def configure_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(levelname)s (%(name)s) %(message)s",
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


def apply_overrides(
    *,
    model: str | None = None,
    max_cycles: int | None = None,
    max_parallel_tasks: int | None = None,
    log_level: str | None = None,
) -> None:
    if model:
        settings.model = model
        settings.worker_model = model
        settings.planner_model = model
        settings.tester_model = model
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel_tasks is not None:
        settings.max_parallel_tasks = max_parallel_tasks
    if log_level:
        settings.log_level = log_level


@click.group()
def main() -> None:
    pass


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model (sets all model fields).")
@click.option(
    "--max-cycles",
    default=None,
    type=int,
    help="Maximum number of planning cycles (0 = unlimited).",
)
@click.option(
    "--max-parallel-tasks",
    default=None,
    type=int,
    help="Maximum number of tasks to run in parallel.",
)
@click.option(
    "--log-level",
    default=None,
    help="Logging level (DEBUG, INFO, WARNING, ERROR).",
)
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel_tasks: int | None,
    log_level: str | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    apply_overrides(
        model=model,
        max_cycles=max_cycles,
        max_parallel_tasks=max_parallel_tasks,
        log_level=log_level,
    )
    configure_logging(settings.log_level)
    log.info(
        "start",
        goal=goal,
        model=settings.model,
        planner_model=settings.planner_model,
        worker_model=settings.worker_model,
        tester_model=settings.tester_model,
        max_cycles=settings.max_cycles,
        max_parallel_tasks=settings.max_parallel_tasks,
        log_level=settings.log_level,
    )
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        log.error("run_failed", error=str(e))
        raise


@main.command()
def web() -> None:
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
