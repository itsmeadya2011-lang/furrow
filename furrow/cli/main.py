"""Command line interface for Furrow.

Commands:
    furrow init            initialize a new Furrow project
    furrow start "<goal>"  start (or resume) an agent run against a goal
    furrow status          show the persisted state of the last run
    furrow resume          continue the persisted run
    furrow history         show run history from persisted state
    furrow config          view or modify Furrow settings
    furrow cost            show detailed cost breakdown
    furrow cancel          cancel a running web server task
    furrow web             serve the web UI
"""

from __future__ import annotations

import asyncio
import inspect
import json
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
        click.echo(f"  Estimated cost:  ${usage.cost_usd:.6f}")
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
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose output (INFO level).")
@click.option("--debug", is_flag=True, help="Enable debug output (DEBUG level).")
@click.pass_context
def main(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """Furrow - an autonomous coding agent."""
    if debug:
        log_level = "DEBUG"
    elif verbose:
        log_level = "INFO"
    else:
        log_level = settings.log_level
    configure_logging(log_level)
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


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


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--force", is_flag=True, help="Overwrite existing configuration.")
def init(workspace: Path | None, force: bool) -> None:
    """Initialize a new Furrow project with default configuration."""
    ws = Path(workspace) if workspace is not None else Path.cwd()
    furrow_dir = ws / ".furrow"
    config_file = furrow_dir / "config.json"
    env_file = ws / ".env.example"

    if config_file.exists() and not force:
        click.secho(f"Configuration already exists at {config_file}", fg="yellow")
        click.echo("Use --force to overwrite.")
        return

    furrow_dir.mkdir(parents=True, exist_ok=True)

    # Write default config
    default_config = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "planner_model": "claude-3-5-haiku-20241022",
        "worker_model": "claude-3-5-sonnet-20241022",
        "tester_model": "claude-3-5-sonnet-20241022",
        "max_parallel_tasks": 5,
        "max_cycles": 0,
        "max_worker_iterations": 12,
        "max_retries": 3,
        "max_task_retries": 1,
        "request_timeout": 120.0,
        "command_timeout": 300.0,
        "test_timeout": 600.0,
        "max_tokens": 4096,
        "log_level": "INFO",
    }
    config_file.write_text(json.dumps(default_config, indent=2))

    # Write .env.example
    env_content = """# Furrow Configuration
# Copy this file to .env and fill in your API keys

# Anthropic (default provider)
ANTHROPIC_API_KEY=your-anthropic-key-here

# OpenAI (alternative provider)
# OPENAI_API_KEY=your-openai-key-here

# Ollama (local models)
# OLLAMA_BASE_URL=http://localhost:11434

# Optional: Override default models
# FURROW_MODEL=claude-sonnet-4-20250514
# FURROW_PLANNER_MODEL=claude-3-5-haiku-20241022
# FURROW_WORKER_MODEL=claude-3-5-sonnet-20241022
# FURROW_TESTER_MODEL=claude-3-5-sonnet-20241022

# Optional: Override other settings
# FURROW_MAX_PARALLEL_TASKS=5
# FURROW_MAX_CYCLES=0
# FURROW_LOG_LEVEL=INFO
"""
    env_file.write_text(env_content)

    click.secho(f"Initialized Furrow project in {ws}", fg="green")
    click.echo(f"  Config: {config_file}")
    click.echo(f"  Env template: {env_file}")
    click.echo("\nNext steps:")
    click.echo("  1. Copy .env.example to .env and add your API keys")
    click.echo("  2. Run 'furrow start \"your goal\"' to begin")


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
def config(workspace: Path | None) -> None:
    """Show the current Furrow configuration."""
    run_settings = _run_settings(workspace=workspace)
    click.echo("Furrow Configuration")
    click.echo("=" * 40)
    for key, value in run_settings.model_dump().items():
        # Mask API keys
        if "api_key" in key.lower() and value:
            value = "***" + str(value)[-4:] if len(str(value)) > 4 else "****"
        click.echo(f"  {key}: {value}")


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--limit", type=int, default=10, help="Number of history entries to show.")
def history(workspace: Path | None, limit: int) -> None:
    """Show run history from persisted state."""
    run_settings = _run_settings(workspace=workspace)
    store = StateStore(run_settings.state_path())
    state = store.load()
    if state is None:
        click.echo("No furrow run state found.")
        return

    click.echo(f"Run History (last {limit} cycles)")
    click.echo("=" * 60)
    click.echo(f"Original goal: {state.original_goal}")
    click.echo(f"Total cycles:  {state.cycles}")
    click.echo()

    history_entries = list(state.history)[-limit:]
    if not history_entries:
        click.echo("No history entries found.")
        return

    for entry in history_entries:
        click.echo(f"Cycle {entry.cycle}:")
        click.echo(f"  Goal:        {entry.goal[:100]}{'...' if len(entry.goal) > 100 else ''}")
        rationale = entry.plan_rationale[:100]
        rationale += "..." if len(entry.plan_rationale) > 100 else ""
        click.echo(f"  Rationale:   {rationale}")
        click.echo(f"  Tasks:       {len(entry.task_summaries)}")
        passed = entry.tests_passed
        verdict = (
            "PASSED" if passed else "FAILED" if passed is not None else "N/A"
        )
        click.echo(f"  Tests:       {verdict}")
        click.echo(f"  Timestamp:   {entry.timestamp}")
        if entry.usage.calls > 0:
            usage = entry.usage
            click.echo(
                f"  Usage:       {usage.calls} calls, {usage.total_tokens} tokens "
                f"(${usage.cost_usd:.6f})"
            )
        click.echo()


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
def cost(workspace: Path | None) -> None:
    """Show detailed cost breakdown from persisted state."""
    run_settings = _run_settings(workspace=workspace)
    store = StateStore(run_settings.state_path())
    state = store.load()
    if state is None:
        click.echo("No furrow run state found.")
        return

    click.echo("Cost Breakdown")
    click.echo("=" * 60)
    click.echo(f"Original goal: {state.original_goal}")
    click.echo(f"Total cycles:  {state.cycles}")
    click.echo()

    total_usage = state.usage
    click.echo("Total:")
    click.echo(f"  Calls:           {total_usage.calls}")
    click.echo(f"  Input tokens:    {total_usage.input_tokens:,}")
    click.echo(f"  Output tokens:   {total_usage.output_tokens:,}")
    click.echo(f"  Total tokens:    {total_usage.total_tokens:,}")
    click.echo(f"  Estimated cost:  ${total_usage.cost_usd:.6f}")
    click.echo()

    history_entries = list(state.history)
    if not history_entries:
        click.echo("No history entries found.")
        return

    click.echo("Per Cycle:")
    click.echo("-" * 60)
    for entry in history_entries:
        if entry.usage.calls == 0:
            continue
        click.echo(f"Cycle {entry.cycle}:")
        click.echo(f"  Calls:           {entry.usage.calls}")
        click.echo(f"  Input tokens:    {entry.usage.input_tokens:,}")
        click.echo(f"  Output tokens:   {entry.usage.output_tokens:,}")
        click.echo(f"  Total tokens:    {entry.usage.total_tokens:,}")
        click.echo(f"  Estimated cost:  ${entry.usage.cost_usd:.6f}")
        click.echo()


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--host", default="127.0.0.1", help="Web server host.")
@click.option("--port", type=int, default=8000, help="Web server port.")
def cancel(workspace: Path | None, host: str, port: int) -> None:
    """Cancel a running Furrow web server task."""
    import httpx

    url = f"http://{host}:{port}/api/cancel"
    try:
        response = httpx.post(url, timeout=5.0)
        if response.status_code == 200:
            click.secho("Run cancelled successfully.", fg="green")
        elif response.status_code == 409:
            click.secho("No run in progress to cancel.", fg="yellow")
        else:
            click.secho(f"Failed to cancel: {response.text}", fg="red")
            sys.exit(1)
    except httpx.ConnectError:
        click.secho(f"Could not connect to web server at {host}:{port}", fg="red")
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Error: {exc}", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
