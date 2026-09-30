from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

import click
import structlog
from rich.console import Console

from furrow.config import Provider, settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def _configure_logging() -> None:
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(level=level, format="%(message)s")
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger("furrow.cli")


def _apply_settings(
    *,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: Path | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    if model:
        settings.model = model
        settings.planner_model = model
        settings.worker_model = model
        settings.tester_model = model
    if planner_model:
        settings.planner_model = planner_model
    if worker_model:
        settings.worker_model = worker_model
    if tester_model:
        settings.tester_model = tester_model
    if provider:
        settings.provider = Provider(provider)
    if max_cycles is not None:
        settings.max_cycles = max_cycles
    if max_parallel is not None:
        settings.max_parallel_tasks = max_parallel
    if workspace is not None:
        settings.workspace = workspace


def _log_settings() -> None:
    dump = {
        k: v for k, v in settings.model_dump(mode="json").items() if not k.endswith("_api_key")
    }
    log.info("furrow configuration", **dump)


def _on_message(msg: str) -> None:
    log.info("progress", msg=msg)


@click.group()
def main() -> None:
    _configure_logging()


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model (sets all role models).")
@click.option("--planner-model", default=None, help="Override planner LLM model.")
@click.option("--worker-model", default=None, help="Override worker LLM model.")
@click.option("--tester-model", default=None, help="Override tester LLM model.")
@click.option(
    "--provider",
    type=click.Choice([p.value for p in Provider], case_sensitive=False),
    default=None,
    help="LLM provider (anthropic/openai/ollama).",
)
@click.option("--max-cycles", type=int, default=None, help="Maximum planning cycles (0 = infinite).")
@click.option("--max-parallel", "max_parallel", type=int, default=None, help="Maximum parallel tasks.")
@click.option(
    "--workspace",
    "workspace",
    type=click.Path(path_type=Path),
    default=None,
    help="Workspace directory.",
)
def start(
    goal: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
    provider: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: Path | None,
) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    _apply_settings(
        model=model,
        planner_model=planner_model,
        worker_model=worker_model,
        tester_model=tester_model,
        provider=provider,
        max_cycles=max_cycles,
        max_parallel=max_parallel,
        workspace=workspace,
    )
    _log_settings()
    log.info("starting furrow", goal=goal)
    try:
        asyncio.run(Orchestrator(goal=goal, on_message=_on_message).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
def web() -> None:
    log.info("starting web server")
    from furrow.web.server import run

    run()


if __name__ == "__main__":
    main()
