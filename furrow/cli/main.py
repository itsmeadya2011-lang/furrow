"""Command line interface for Furrow.

Commands:
    furrow start "<goal>"  start (or resume) an agent run against a goal
    furrow status          show the persisted state of the last run
    furrow resume          continue the persisted run
    furrow web             serve the web UI
    furrow config          show or modify configuration
    furrow doctor          check the environment and configuration
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
from furrow.exceptions import FurrowError
from furrow.logging import configure_logging
from furrow.security import CommandValidator

logger = logging.getLogger(__name__)

WORKSPACE_HELP = "Project directory the agent works in."
MAX_CYCLES_HELP = "Maximum number of cycles (0 = unlimited)."
PROVIDER_HELP = "LLM provider to use (anthropic, openai, ollama, google, azure, deepseek, groq, mistral, xai)."
MODEL_HELP = "Model to use for the main LLM calls."
LOG_LEVEL_HELP = "Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."


def _run_settings(
    model: str | None = None,
    provider: str | None = None,
    max_cycles: int | None = None,
    workspace: Path | None = None,
    log_level: str | None = None,
    max_parallel_tasks: int | None = None,
    max_worker_iterations: int | None = None,
    request_timeout: float | None = None,
    command_timeout: float | None = None,
    test_timeout: float | None = None,
    max_tokens: int | None = None,
) -> Settings:
    """Return a validated copy of the global settings with the given overrides applied.

    The module level ``settings`` object is never mutated: a fresh validated
    instance is constructed for this run only.
    """
    updates: dict[str, Any] = {}
    if model is not None:
        updates["model"] = model
    if provider is not None:
        updates["provider"] = provider
    if max_cycles is not None:
        updates["max_cycles"] = max_cycles
    if workspace is not None:
        updates["workspace"] = Path(workspace)
    if log_level is not None:
        updates["log_level"] = log_level
    if max_parallel_tasks is not None:
        updates["max_parallel_tasks"] = max_parallel_tasks
    if max_worker_iterations is not None:
        updates["max_worker_iterations"] = max_worker_iterations
    if request_timeout is not None:
        updates["request_timeout"] = request_timeout
    if command_timeout is not None:
        updates["command_timeout"] = command_timeout
    if test_timeout is not None:
        updates["test_timeout"] = test_timeout
    if max_tokens is not None:
        updates["max_tokens"] = max_tokens
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
    click.echo("\n" + click.style("Run summary", bold=True))
    click.echo(f"  Cycles:            {result.cycles}")
    click.echo(f"  Tasks completed:   {result.tasks_completed}")
    click.echo(f"  Tasks failed:      {result.tasks_failed}")
    blocked = getattr(result, "tasks_blocked", 0) or 0
    click.echo(f"  Tasks blocked:     {blocked}")
    click.echo(f"  Stopped reason:    {result.stopped_reason}")
    skipped = bool(getattr(result, "skipped", False))
    tests_run = getattr(result, "tests_run", 0) or 0
    text, colour = _result_line(result.passed, skipped, tests_run)
    click.secho(f"  Result:            {text}", fg=colour)
    usage = getattr(result, "usage", None)
    if usage is not None and usage.calls > 0:
        click.echo(
            f"  LLM usage:         {usage.calls} calls, {usage.total_tokens} tokens "
            f"(in {usage.input_tokens} / out {usage.output_tokens})"
        )
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        if cost > 0:
            click.echo(f"  LLM cost:          ${cost:.6f}")
    files_changed = getattr(result, "files_changed", None) or []
    if files_changed:
        click.echo(f"  Files changed:     {', '.join(files_changed)}")
    if result.summary:
        click.echo(f"  Summary:           {result.summary}")


def _print_state(state: RunState) -> None:
    """Print the persisted run state."""
    click.echo("\n" + click.style("Furrow run status", bold=True))
    click.echo(f"  Original goal:     {state.original_goal}")
    click.echo(f"  Current goal:      {state.goal}")
    click.echo(f"  Cycles:            {state.cycles}")
    click.echo(f"  Updated:           {state.updated_at}")
    tests_run = getattr(state, "tests_run", 0) or 0
    skipped = bool(getattr(state, "last_skipped", False))
    passed = getattr(state, "last_passed", None)
    text, colour = _result_line(passed=passed, skipped=skipped, tests_run=tests_run)
    click.echo(f"  Tests run:         {tests_run}")
    click.secho(f"  Result:            {text}", fg=colour)
    usage = getattr(state, "usage", None)
    if usage is not None and usage.calls > 0:
        click.echo(f"  LLM usage:         {usage.calls} calls, {usage.total_tokens} tokens")
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        if cost > 0:
            click.echo(f"  LLM cost:          ${cost:.6f}")
    if state.tasks:
        click.echo("\n  Tasks:")
        for task in state.tasks:
            files_changed = getattr(task, "files_changed", None) or []
            line = f"    {task.id:>3}  {task.status:<10}  {task.description}"
            if files_changed:
                line += f"  [{', '.join(files_changed)}]"
            click.echo(line)
    else:
        click.echo("  Tasks:             (none)")


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
        click.echo("\nFurrow stopped by user.")
        sys.exit(130)
    except FurrowError as exc:
        click.secho(f"\nFurrow failed: {exc.message}", fg="red")
        if exc.details:
            click.echo(f"Details: {json.dumps(exc.details, indent=2)}")
        sys.exit(1)
    except Exception as exc:
        click.secho(f"\nFurrow failed: {exc}", fg="red")
        sys.exit(1)
    _print_summary(result)
    skipped = bool(getattr(result, "skipped", False))
    tests_run = getattr(result, "tests_run", 0) or 0
    if skipped or result.passed or tests_run == 0:
        sys.exit(0)
    else:
        sys.exit(1)


@click.group()
@click.version_option(package_name="furrow")
def main() -> None:
    """Furrow - an autonomous coding agent with an infinite parallel development loop."""
    configure_logging(settings.log_level)


@main.command()
@click.argument("goal", required=False)
@click.option("--model", default=None, help=MODEL_HELP)
@click.option("--provider", type=click.Choice(["anthropic", "openai", "ollama", "google", "azure", "deepseek", "groq", "mistral", "xai"]), default=None, help=PROVIDER_HELP)
@click.option("--max-cycles", type=click.IntRange(min=0), default=None, help=MAX_CYCLES_HELP)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--resume", is_flag=True, help="Resume the persisted run instead of starting fresh.")
@click.option("--log-level", type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]), default=None, help=LOG_LEVEL_HELP)
@click.option("--max-parallel-tasks", type=click.IntRange(min=1, max=50), default=None, help="Maximum number of parallel tasks.")
@click.option("--max-worker-iterations", type=click.IntRange(min=1, max=100), default=None, help="Maximum iterations per worker task.")
@click.option("--request-timeout", type=float, default=None, help="LLM request timeout in seconds.")
@click.option("--command-timeout", type=float, default=None, help="Shell command timeout in seconds.")
@click.option("--test-timeout", type=float, default=None, help="Test suite timeout in seconds.")
@click.option("--max-tokens", type=click.IntRange(min=1, max=100000), default=None, help="Maximum tokens per LLM response.")
def start(
    goal: str | None,
    model: str | None,
    provider: str | None,
    max_cycles: int | None,
    workspace: Path | None,
    resume: bool,
    log_level: str | None,
    max_parallel_tasks: int | None,
    max_worker_iterations: int | None,
    request_timeout: float | None,
    command_timeout: float | None,
    test_timeout: float | None,
    max_tokens: int | None,
) -> None:
    """Start a Furrow run for GOAL (prompted for when omitted)."""
    if not goal:
        goal = click.prompt("Enter your goal for Furrow")
    run_settings = _run_settings(
        model=model,
        provider=provider,
        max_cycles=max_cycles,
        workspace=workspace,
        log_level=log_level,
        max_parallel_tasks=max_parallel_tasks,
        max_worker_iterations=max_worker_iterations,
        request_timeout=request_timeout,
        command_timeout=command_timeout,
        test_timeout=test_timeout,
        max_tokens=max_tokens,
    )
    _execute(goal=goal, run_settings=run_settings, resume=resume)


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def status(workspace: Path | None, as_json: bool) -> None:
    """Show the state of the persisted Furrow run."""
    run_settings = _run_settings(workspace=workspace)
    store = StateStore(run_settings.state_path())
    state = store.load()
    if state is None:
        click.echo("No furrow run state found.")
        return
    if as_json:
        click.echo(json.dumps(state.model_dump(mode="json"), indent=2))
    else:
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
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def config(workspace: Path | None, as_json: bool) -> None:
    """Show the current configuration."""
    run_settings = _run_settings(workspace=workspace)
    if as_json:
        # Exclude sensitive fields
        data = run_settings.model_dump(mode="json")
        for key in ["anthropic_api_key", "openai_api_key", "google_api_key", "azure_api_key",
                    "deepseek_api_key", "groq_api_key", "mistral_api_key", "xai_api_key"]:
            if key in data:
                data[key] = "***" if data[key] else None
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(click.style("Furrow Configuration", bold=True))
        click.echo(f"  Provider:              {run_settings.provider.value}")
        click.echo(f"  Model:                 {run_settings.model}")
        click.echo(f"  Planner Model:         {run_settings.planner_model}")
        click.echo(f"  Worker Model:          {run_settings.worker_model}")
        click.echo(f"  Tester Model:          {run_settings.tester_model}")
        click.echo(f"  Max Parallel Tasks:    {run_settings.max_parallel_tasks}")
        click.echo(f"  Max Cycles:            {run_settings.max_cycles or 'unlimited'}")
        click.echo(f"  Workspace:             {run_settings.workspace}")
        click.echo(f"  Log Level:             {run_settings.log_level}")
        click.echo(f"  Max Worker Iterations: {run_settings.max_worker_iterations}")
        click.echo(f"  Max Retries:           {run_settings.max_retries}")
        click.echo(f"  Max Task Retries:      {run_settings.max_task_retries}")
        click.echo(f"  Request Timeout:       {run_settings.request_timeout}s")
        click.echo(f"  Command Timeout:       {run_settings.command_timeout}s")
        click.echo(f"  Test Timeout:          {run_settings.test_timeout}s")
        click.echo(f"  Max Tokens:            {run_settings.max_tokens}")
        click.echo(f"  State File:            {run_settings.state_file}")
        click.echo(f"  Web Host:              {run_settings.web.host}")
        click.echo(f"  Web Port:              {run_settings.web.port}")
        click.echo(f"  Web CORS:              {run_settings.web.enable_cors}")


@main.command()
def doctor() -> None:
    """Check the environment and configuration for common issues."""
    click.echo(click.style("Furrow Doctor", bold=True))
    click.echo()

    issues = []
    warnings = []

    # Check Python version
    import sys
    if sys.version_info < (3, 10):
        issues.append(f"Python 3.10+ required, found {sys.version}")
    else:
        click.secho(f"  ✓ Python {sys.version.split()[0]}", fg="green")

    # Check API keys
    provider = settings.provider
    if provider.value == "anthropic":
        if not settings.anthropic_api_key and not __import__("os").getenv("ANTHROPIC_API_KEY"):
            issues.append("ANTHROPIC_API_KEY not set")
        else:
            click.secho("  ✓ ANTHROPIC_API_KEY set", fg="green")
    elif provider.value == "openai":
        if not settings.openai_api_key and not __import__("os").getenv("OPENAI_API_KEY"):
            issues.append("OPENAI_API_KEY not set")
        else:
            click.secho("  ✓ OPENAI_API_KEY set", fg="green")
    elif provider.value == "ollama":
        click.secho(f"  ✓ Using Ollama at {settings.ollama_base_url}", fg="green")

    # Check workspace
    if not settings.workspace.exists():
        warnings.append(f"Workspace does not exist: {settings.workspace}")
    else:
        click.secho(f"  ✓ Workspace exists: {settings.workspace}", fg="green")

    # Check state directory
    state_path = settings.state_path()
    if not state_path.parent.exists():
        warnings.append(f"State directory does not exist: {state_path.parent}")
    else:
        click.secho(f"  ✓ State directory exists: {state_path.parent}", fg="green")

    # Check for test runners
    test_runners = []
    if (settings.workspace / "pyproject.toml").exists():
        test_runners.append("pytest (pyproject.toml)")
    if (settings.workspace / "pytest.ini").exists():
        test_runners.append("pytest (pytest.ini)")
    if (settings.workspace / "package.json").exists():
        test_runners.append("npm/pnpm/yarn test")
    if (settings.workspace / "Cargo.toml").exists():
        test_runners.append("cargo test")
    if (settings.workspace / "go.mod").exists():
        click.secho("  ✓ Go project detected", fg="green")
    if test_runners:
        click.secho(f"  ✓ Test runners detected: {', '.join(test_runners)}", fg="green")
    else:
        warnings.append("No test runner detected (tests will be skipped)")

    # Check dependencies
    try:
        import anthropic
        click.secho("  ✓ anthropic available", fg="green")
    except ImportError:
        if provider.value == "anthropic":
            issues.append("anthropic package not installed")
        else:
            warnings.append("anthropic package not installed")

    try:
        import openai
        click.secho("  ✓ openai available", fg="green")
    except ImportError:
        if provider.value == "openai":
            issues.append("openai package not installed")
        else:
            warnings.append("openai package not installed")

    # Summary
    click.echo()
    if warnings:
        click.secho("Warnings:", bold=True, fg="yellow")
        for w in warnings:
            click.secho(f"  ! {w}", fg="yellow")
    if issues:
        click.secho("Issues:", bold=True, fg="red")
        for i in issues:
            click.secho(f"  ✗ {i}", fg="red")
        sys.exit(1)
    else:
        click.secho("All checks passed!", fg="green")


@main.command()
@click.argument("command", required=True)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
def validate_command(command: str, workspace: Path | None) -> None:
    """Validate a shell command against the security policy."""
    run_settings = _run_settings(workspace=workspace)
    workspace_path = run_settings.workspace
    validator = CommandValidator(
        workspace_path,
        denied_commands=run_settings.get_denied_commands(),
        denied_programs=run_settings.get_denied_programs(),
    )
    denied = validator.is_command_denied(command)
    if denied:
        click.secho("DENIED", fg="red", bold=True)
        click.echo(f"Command: {command}")
        sys.exit(1)
    else:
        click.secho("ALLOWED", fg="green", bold=True)
        click.echo(f"Command: {command}")


if __name__ == "__main__":
    main()