from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

import click
from rich.console import Console

from furrow.config import settings
from furrow.core.orchestrator import Orchestrator

console = Console()


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _apply_overrides(
    *,
    provider: str | None = None,
    model: str | None = None,
    planner_model: str | None = None,
    worker_model: str | None = None,
    tester_model: str | None = None,
    max_cycles: int | None = None,
    workspace: str | None = None,
) -> None:
    overrides: dict[str, Any] = {}
    if provider:
        overrides["provider"] = provider
    if model:
        overrides["model"] = model
    if planner_model:
        overrides["planner_model"] = planner_model
    if worker_model:
        overrides["worker_model"] = worker_model
    if tester_model:
        overrides["tester_model"] = tester_model
    if max_cycles is not None:
        overrides["max_cycles"] = max_cycles
    if workspace:
        overrides["workspace"] = os.path.abspath(os.path.expanduser(workspace))

    for key, value in overrides.items():
        if hasattr(settings, key):
            setattr(settings, key, value)
        else:
            console.print(f"[yellow]Warning: unknown setting '{key}' ignored.[/yellow]")


@click.group(help="Furrow - autonomous coding agent CLI.")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _configure_logging(verbose)


@main.command(help="Start the Furrow orchestrator to work on a coding goal.")
@click.option(
    "-c",
    "--cycles",
    type=int,
    default=None,
    help="Override the maximum number of orchestrator cycles (settings.max_cycles).",
)
@click.option(
    "-w",
    "--workspace",
    type=click.Path(file_okay=False, dir_okay=True),
    default=None,
    help="Set the workspace directory (defaults to settings.workspace).",
)
@click.option(
    "--provider",
    type=click.Choice(["anthropic", "openai", "ollama"], case_sensitive=False),
    default=None,
    help="Override the LLM provider (anthropic, openai, ollama).",
)
@click.option(
    "--model",
    default=None,
    help="Override the default LLM model.",
)
@click.option(
    "--planner-model",
    default=None,
    help="Override the model used by the planner agent.",
)
@click.option(
    "--worker-model",
    default=None,
    help="Override the model used by the worker agent.",
)
@click.option(
    "--tester-model",
    default=None,
    help="Override the model used by the tester agent.",
)
@click.argument("goal", required=False)
@click.pass_context
def start(
    ctx: click.Context,
    goal: str | None,
    cycles: int | None,
    workspace: str | None,
    provider: str | None,
    model: str | None,
    planner_model: str | None,
    worker_model: str | None,
    tester_model: str | None,
) -> None:
    """Start the Furrow orchestrator to work on a coding GOAL."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")

    _apply_overrides(
        provider=provider,
        model=model,
        planner_model=planner_model,
        worker_model=worker_model,
        tester_model=tester_model,
        max_cycles=cycles,
        workspace=workspace,
    )

    if workspace:
        os.chdir(settings.workspace)

    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command(help="Launch the Furrow web UI server.")
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="Host interface to bind the web server to.",
)
@click.option(
    "--port",
    type=int,
    default=8000,
    show_default=True,
    help="Port to bind the web server to.",
)
@click.option(
    "--reload/--no-reload",
    default=False,
    show_default=True,
    help="Enable auto-reload (development mode).",
)
def web(host: str, port: int, reload: bool) -> None:
    """Launch the Furrow web UI server."""
    from furrow.web.server import run

    console.print(f"[cyan]Starting Furrow web UI on http://{host}:{port}[/cyan]")
    run(host=host, port=port, reload=reload)


@main.command(help="Print the currently effective Furrow settings.")
def config() -> None:
    """Print the currently effective Furrow settings."""
    import json

    data = {k: getattr(settings, k) for k in sorted(vars(settings))}
    console.print_json(json.dumps(data, default=str, indent=2))


if __name__ == "__main__":
    main()