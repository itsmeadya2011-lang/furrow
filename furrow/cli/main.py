"""Command line interface for Furrow.

Commands:
    furrow start "<goal>"  start (or resume) an agent run against a goal
    furrow status          show the persisted state of the last run
    furrow resume          continue the persisted run
    furrow web             serve the web UI
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from pathlib import Path
from typing import Any

import click

from furrow.config import RunResult, Settings, settings
from furrow.core.orchestrator import Orchestrator
from furrow.core.state import RunState, StateStore
from furrow.logging import configure_logging

logger = logging.getLogger(__name__)

WORKSPACE_HELP = "Project directory the agent works in."
MAX_CYCLES_HELP = "Maximum number of cycles (0 = unlimited)."


def _run_settings(
    model: str | None = None,
    max_cycles: int | None = None,
    workspace: Path | None = None,
) -> Settings:
    """Return a validated copy of the global settings with the given overrides applied.

    The module level ``settings`` object is never mutated: a fresh validated
    instance is constructed for this run only.
    """
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if workspace is not None:
        updates["workspace"] = Path(workspace)
    merged = {**settings.model_dump(), **updates}
    return Settings.model_validate(merged)


def _result_line(passed: bool | None, skipped: bool, tests_run: int) -> tuple[str, str]:
    """Return the (verdict, colour) of the one-line result of a run."""
    if skipped:
        return ("SKIPPED - no test runner detected.", "yellow")
    if passed:
        return ("PASSED - all tests passed.", "green")
    if tests_run == 0:
        return ("NOT RUN - no tests were executed.", "yellow")
    return ("FAILED - tests did not pass.", "red")


def _print_summary(result: RunResult) -> None:
    """Print the summary of a finished run."""
    click.echo("Run summary")
    click.echo(f"  Cycles:          {result.cycles}")
    click.echo(f"  Tasks completed: {result.tasks_completed}")
    click.echo(f"  Tasks failed:    {result.tasks_failed}")
    blocked = getattr(result, "tasks_blocked", 0) or 0
    click.echo(f"  Tasks blocked:  {blocked}")
    click.echo(f"  Stopped reason:  {result.stopped_reason}")
    skipped = bool(getattr(result, "skipped", False))
    tests_run = getattr(result, "tests_run", 0) or 0
    text, colour = _result_line(result.passed, skipped, tests_run)
    click.secho(f"  Result: {text}", fg=colour)
    usage = getattr(result, "usage", None)
    if usage is not None and usage.calls > 0:
        click.echo(
            f"  LLM usage:       {usage.calls} calls, {usage.total_tokens} tokens "
            f"(in {usage.input_tokens} / out {usage.output_tokens})"
        )
        if usage.cost_usd > 0:
            click.echo(f"  LLM cost:        ${usage.cost_usd:.6f}")
    files_changed = getattr(result, "files_changed", None) or []
    if files_changed:
        click.echo(f"  Files changed:    {', '.join(files_changed)}")
    if result.summary:
        click.echo(f"  Summary: {result.summary}")


def _print_state(state: RunState) -> None:
    """Print the persisted run state."""
    click.echo("Furrow run status")
    click.echo(f"  Original goal: {state.original_goal}")
    click.echo(f"  Current goal:  {state.goal}")
    click.echo(f"  Cycles:        {state.cycles}")
    click.echo(f"  Updated:       {state.updated_at}")
    tests_run = getattr(state, "tests_run", 0) or 0
    skipped = bool(getattr(state, "last_skipped", False))
    passed = getattr(state, "last_passed", None)
    text, colour = _result_line(passed=passed, skipped=skipped, tests_run=tests_run)
    click.echo(f"  Tests run:     {tests_run}")
    click.secho(f"  Result:        {text}", fg=colour)
    usage = getattr(state, "usage", None)
    if usage is not None and usage.calls > 0:
        click.echo(f"  LLM usage:       {usage.calls} calls, {usage.total_tokens} tokens")
    if state.tasks:
        click.echo("Tasks:")
        for task in state.tasks:
            files_changed = getattr(task, "files_changed", None) or []
            line = f"  {task.id}  {task.status}  {task.description}"
            if files_changed:
                line += f"  [{', '.join(files_changed)}]"
            click.echo(line)
    else:
        click.echo("Tasks:  (none)")


async def _release_client(client: Any) -> None:
    """Release the HTTP resources of an LLM client, when it supports closing.

    ``LLMClient.aclose()`` may not exist yet and may be a plain or an async
    method, so it is looked up defensively and only awaited when the result
    is awaitable.  A failing close is logged and ignored: closing must never
    turn a successful run into a failed one.
    """
    closer = getattr(client, "aclose", None)
    if closer is None:
        return
    try:
        outcome = closer()
        if inspect.isawaitable(outcome):
            await outcome
    except Exception as exc:
        logger.warning("Could not close the LLM client: %s", exc)


async def _run_and_release(orchestrator: Orchestrator) -> RunResult:
    """Run an orchestrator and release its LLM client afterwards.

    The client is closed in a ``finally`` block, so its HTTP connections are
    released even when the run raises.
    """
    try:
        return await orchestrator.run()
    finally:
        await _release_client(getattr(orchestrator, "client", None))


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
        result = asyncio.run(_run_and_release(orchestrator))
    except KeyboardInterrupt:
        click.echo("Furrow stopped by user.")
        sys.exit(0)
    except Exception as exc:
        click.secho(f"Furrow failed: {exc}", fg="red")
        sys.exit(1)
    _print_summary(result)
    skipped = bool(getattr(result, "skipped", False))
    tests_run = getattr(result, "tests_run", 0) or 0
    if skipped or result.passed or tests_run == 0:
        sys.exit(0)
    else:
        sys.exit(1)


@click.group()
def main() -> None:
    """Furrow - an autonomous coding agent."""
    configure_logging(settings.log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help="Override the LLM model.")
@click.option("--max-cycles", type=click.IntRange(min=0), default=None, help=MAX_CYCLES_HELP)
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
@click.option("--max-cycles", type=click.IntRange(min=0), default=None, help=MAX_CYCLES_HELP)
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

    try:
        run(host=host, port=port)
    except Exception as exc:
        click.secho(f"Furrow failed: {exc}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
