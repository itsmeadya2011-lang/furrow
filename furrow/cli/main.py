from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import click
from rich.console import Console

from furrow.core.orchestrator import Orchestrator

console = Console()


def _configure_logging() -> None:
    """Apply the configured log level if the config module exposes a helper."""
    from furrow.config import settings

    module = importlib.import_module("furrow.config")
    helper = getattr(module, "configure_logging", None)
    if helper:
        helper(settings.log_level)


def _set_model(model: str | None) -> None:
    """Override the LLM model, failing loudly instead of at first LLM call."""
    if model is None:
        return
    name = model.strip()
    if not name:
        raise click.BadParameter("model must not be empty", param_hint="--model")
    from furrow.config import settings

    if not hasattr(settings, "model"):
        raise click.ClickException("this build of furrow has no configurable model setting")
    try:
        settings.model = name
    except (AttributeError, ValueError) as exc:
        raise click.ClickException(f"invalid model {name!r}: {exc}") from exc


def _set_workspace(workspace: str | None) -> None:
    """Point the run at a specific workspace directory."""
    if workspace is None:
        return
    from furrow.config import settings

    resolved = Path(workspace).expanduser().resolve()
    if not hasattr(settings, "workspace"):
        raise click.ClickException("this build of furrow has no configurable workspace setting")
    settings.workspace = resolved


def _set_cycles(cycles: int | None) -> None:
    """Cap the number of planning/execution cycles."""
    if cycles is None:
        return
    from furrow.config import settings

    if cycles < 0:
        raise click.BadParameter("cycles must be zero or more", param_hint="--cycles")
    if not hasattr(settings, "max_cycles"):
        raise click.ClickException("this build of furrow has no configurable cycle limit")
    settings.max_cycles = cycles


@click.group()
def main() -> None:
    """Furrow: an autonomous coding agent."""


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override LLM model")
@click.option("--cycles", default=None, type=int, help="Maximum cycles (0 = unlimited)")
@click.option("--workspace", default=None, help="Workspace directory for the run")
def start(goal: str | None, model: str | None, cycles: int | None, workspace: str | None) -> None:
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    _set_model(model)
    _set_workspace(workspace)
    _set_cycles(cycles)
    _configure_logging()
    try:
        asyncio.run(Orchestrator(goal=goal).run())
    except KeyboardInterrupt:
        console.print("\n[yellow]Furrow stopped by user.[/yellow]")
        sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Bind address")
@click.option("--port", default=8000, show_default=True, type=int, help="Bind port")
def web(host: str, port: int) -> None:
    if not 0 < port < 65536:
        raise click.BadParameter("port must be between 1 and 65535", param_hint="--port")
    from furrow.web.server import run

    console.print(f"[bold green]Furrow[/bold green] web UI on http://{host}:{port}")
    run(host, port)


if __name__ == "__main__":
    main()