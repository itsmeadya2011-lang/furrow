"""Command line interface for Furrow.

Commands:
    furrow start "<goal>"  start (or resume) an agent run against a goal
    furrow status          show the persisted state of the last run
    furrow resume          continue the persisted run
    furrow web             serve the web UI
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from furrow.config import RunResult, Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import RunState, StateStore
from furrow.logging import configure_logging

WORKSPACE_HELP = "Project directory the agent works in."
MAX_CYCLES_HELP = "Maximum number of cycles (0 = unlimited)."


def _run_settings(
    model: str | None = None,
    max_cycles: int | None = None,
    workspace: Path | None = None,
) -> Settings:
    """Return a copy of the global settings with the given overrides applied.

    The module level ``settings`` object is never mutated: ``model_copy`` gives
    us a fresh instance for this run only.
    """
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if workspace is not None:
        updates["workspace"] = Path(workspace)
    return settings.model_copy(update=updates)


def _print_summary(result: RunResult) -> None:
    """Print the summary of a finished run."""
    click.echo("Run summary")
    click.echo(f"  Cycles:          {result.cycles}")
    click.echo(f"  Tasks completed: {result.tasks_completed}")
    click.echo(f"  Tasks failed:    {result.tasks_failed}")
    click.echo(f"  Stopped reason:  {result.stopped_reason}")
    skipped = bool(getattr(result, "skipped", False))
    if skipped:
        click.secho("  Result: SKIPPED - no test runner detected.", fg="yellow")
    elif result.passed:
        click.secho("  Result: PASSED - all tests passed.", fg="green")
    else:
        click.secho("  Result: FAILED - tests did not pass.", fg="red")
    if result.summary:
        click.echo(f"  Summary: {result.summary}")


def _print_state(state: RunState) -> None:
    """Print the persisted run state."""
    click.echo("Furrow run status")
    click.echo(f"  Original goal: {state.original_goal}")
    click.echo(f"  Current goal:  {state.goal}")
    click.echo(f"  Cycles:        {state.cycles}")
    click.echo(f"  Updated:       {state.updated_at}")
    if state.tasks:
        click.echo("Tasks:")
        for task in state.tasks:
            click.echo(f"  {task.id}  {task.status}  {task.description}")
    else:
        click.echo("Tasks:  (none)")


def _execute(goal: str, run_settings: Settings, resume: bool) -> None:
    """Run the orchestrator and print the summary of the result."""
    store = StateStore(run_settings.state_path())
    orchestrator = Orchestrator(
        goal=goal,
        settings=run_settings,
        state_store=store,
        resume=resume,
    )
    try:
        result = asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        click.echo("Furrow stopped by user.")
        sys.exit(0)
    except Exception as exc:
        click.secho(f"Furrow failed: {exc}", fg="red")
        sys.exit(1)
    _print_summary(result)


@click.group()
def main() -> None:
    """Furrow - an autonomous coding agent."""
    configure_logging(settings.log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the LLM model.")
@click.option("--max-cycles", type=int, default=None, help=MAX_CYCLES_HELP)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--resume", is_flag=True, help="Resume the persisted run instead of starting fresh.")
def start(
    goal: str | None,
    model: str | None,
    max_cycles: int | None,
    workspace: Path | None,
    resume: bool,
) -> None:
    """Start a Furrow run for GOAL (prompted for when omitted)."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    run_settings = _run_settings(model=model, max_cycles=max_cycles, workspace=workspace)
    _execute(goal=goal, run_settings=run_settings, resume=resume)


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
def status(workspace: Path | None) -> None:
    """Show the state of the persisted Furrow run."""
    run_settings = _run_settings(workspace=workspace)
    store = StateStore(run_settings.state_path())
    state = store.load()
    if state is None:
        click.echo("No furrow run state found.")
        return
    _print_state(state)


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--max-cycles", type=int, default=None, help=MAX_CYCLES_HELP)
def resume(workspace: Path | None, max_cycles: int | None) -> None:
    """Resume the persisted Furrow run."""
    run_settings = _run_settings(max_cycles=max_cycles, workspace=workspace)
    store = StateStore(run_settings.state_path())
    state = store.load()
    if state is None:
        click.secho("No furrow run state found to resume.", fg="red")
        sys.exit(1)
    goal = state.original_goal or state.goal
    _execute(goal=goal, run_settings=run_settings, resume=True)


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the web server to.")
@click.option("--port", type=int, default=8000, help="Port to bind the web server to.")
def web(host: str, port: int) -> None:
    """Run the Furrow web server."""
    from furrow.web.server import run

    run(host=host, port=port)


if __name__ == "__main__":
    main()
