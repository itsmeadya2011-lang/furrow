"""Command line interface for Furrow.

Commands:
    furrow start "<goal>"  start (or resume) an agent run against a goal
    furrow status          show the persisted state of the last run
    furrow resume          continue the persisted run
    furrow web             serve the web UI
    furrow doctor          check the environment and configuration
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import inspect
import logging
import os
import shlex
import shutil
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

# Key third-party packages Furrow depends on, mapped to their import names.
_REQUIRED_PACKAGES = {
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "pydantic-settings": "pydantic_settings",
    "anthropic": "anthropic",
    "openai": "openai",
    "httpx": "httpx",
    "click": "click",
    "rich": "rich",
    "python-dotenv": "dotenv",
    "aiofiles": "aiofiles",
    "tenacity": "tenacity",
    "structlog": "structlog",
}

# Test runners the worker/tester can invoke, mapped to the binary to look for.
_TEST_RUNNERS = {
    "pytest": ["pytest"],
    "npm": ["npm"],
    "pnpm": ["pnpm"],
    "yarn": ["yarn"],
    "cargo": ["cargo"],
    "go": ["go"],
}


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
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        if cost > 0:
            click.echo(f"  LLM cost:        ${cost:.6f}")
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
        cost = float(getattr(usage, "cost_usd", 0.0) or 0.0)
        if cost > 0:
            click.echo(f"  LLM cost:        ${cost:.6f}")
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
    except KeyboardInterrupt:
        raise
    except SystemExit as exc:
        message = str(exc.code) if exc.code is not None else "the server could not start"
        click.secho(f"Furrow failed: {message}", fg="red")
        sys.exit(1)
    except Exception as exc:
        click.secho(f"Furrow failed: {exc}", fg="red")
        sys.exit(1)


def _check_package(name: str) -> tuple[str, str | None]:
    """Return (status, version) for a package: ``("ok", version)`` or ``("missing", None)``.

    ``name`` is the distribution/package name as registered with pip.
    """
    try:
        version = importlib.metadata.version(name)
        return ("ok", version)
    except importlib.metadata.PackageNotFoundError:
        return ("missing", None)


def _check_binary(name: str) -> bool:
    """Return True when *name* resolves to an executable on PATH."""
    return shutil.which(name) is not None


def _check_env_var(name: str) -> tuple[str, str]:
    """Return (status, value) for an environment variable."""
    value = os.getenv(name)
    if value:
        # Show only the last 4 characters of a secret value.
        masked = "***" + value[-4:] if len(value) > 4 else "****"
        return ("set", masked)
    return ("unset", "")


def _doctor_workspace(workspace: Path) -> list[str]:
    """Return a list of human-readable lines describing the workspace."""
    lines: list[str] = []
    if not workspace.exists():
        lines.append(f"  Workspace: {workspace} (does not exist)")
        return lines
    if not workspace.is_dir():
        lines.append(f"  Workspace: {workspace} (not a directory)")
        return lines
    lines.append(f"  Workspace: {workspace}")
    markers = ["pyproject.toml", "package.json", "Cargo.toml", "go.mod", "requirements.txt",
               "setup.cfg", "tox.ini", "pytest.ini"]
    found = [m for m in markers if (workspace / m).is_file()]
    if found:
        lines.append(f"  Project markers: {', '.join(found)}")
    tests_dir = workspace / "tests"
    if tests_dir.is_dir():
        test_files = [p.name for p in tests_dir.iterdir() if p.is_file() and p.suffix == ".py"]
        lines.append(f"  tests/ directory: {len(test_files)} test file(s)")
    state_path = workspace / ".furrow" / "state.json"
    if state_path.is_file():
        lines.append(f"  State file: {state_path} (exists)")
    else:
        lines.append(f"  State file: {state_path} (not found)")
    return lines


@main.command()
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help=WORKSPACE_HELP,
)
def doctor(workspace: Path | None) -> None:
    """Check the environment, dependencies, and configuration.

    Reports Python version, installed packages, API keys, provider settings,
    detected test runners, and workspace state. Exits 0 when everything looks
    sane, non-zero when a problem is found.
    """
    run_settings = _run_settings(workspace=workspace)
    ws = run_settings.workspace
    problems: list[str] = []

    click.echo("Furrow doctor")
    click.echo("=" * 40)

    # Python version
    py = sys.version.split()[0]
    click.echo(f"Python: {py}")
    if sys.version_info < (3, 10):
        problems.append(f"Python {py} is below the required 3.10")

    # Furrow version
    try:
        furrow_version = importlib.metadata.version("furrow")
        click.echo(f"Furrow: {furrow_version}")
    except importlib.metadata.PackageNotFoundError:
        click.echo("Furrow: (not installed via pip)")
        problems.append("Furrow is not installed as a package")

    click.echo()
    click.echo("Dependencies:")
    all_ok = True
    for name in _REQUIRED_PACKAGES:
        status, version = _check_package(name)
        marker = "✓" if status == "ok" else "✗"
        version_str = version if version else "not installed"
        click.echo(f"  {marker} {name}: {version_str}")
        if status != "ok":
            all_ok = False
            problems.append(f"Missing dependency: {name}")
    if not all_ok:
        click.echo("  Install missing dependencies with: pip install -e \".[dev]\"")

    click.echo()
    click.echo("LLM provider:")
    click.echo(f"  Provider: {run_settings.provider.value}")
    click.echo(f"  Model: {run_settings.model}")
    click.echo(f"  Planner model: {run_settings.planner_model}")
    click.echo(f"  Worker model: {run_settings.worker_model}")
    click.echo(f"  Tester model: {run_settings.tester_model}")
    if run_settings.provider.value == "anthropic":
        status, masked = _check_env_var("FURROW_ANTHROPIC_API_KEY")
        if status == "unset":
            status, masked = _check_env_var("ANTHROPIC_API_KEY")
        click.echo(f"  Anthropic API key: {status} ({masked or 'not set'})")
        if status == "unset":
            problems.append("No Anthropic API key found (set FURROW_ANTHROPIC_API_KEY or ANTHROPIC_API_KEY)")
    elif run_settings.provider.value == "openai":
        status, masked = _check_env_var("FURROW_OPENAI_API_KEY")
        if status == "unset":
            status, masked = _check_env_var("OPENAI_API_KEY")
        click.echo(f"  OpenAI API key: {status} ({masked or 'not set'})")
        if status == "unset":
            problems.append("No OpenAI API key found (set FURROW_OPENAI_API_KEY or OPENAI_API_KEY)")
    elif run_settings.provider.value == "ollama":
        click.echo(f"  Ollama base URL: {run_settings.ollama_base_url}")

    click.echo()
    click.echo("Settings:")
    click.echo(f"  Max parallel tasks: {run_settings.max_parallel_tasks}")
    click.echo(f"  Max cycles: {run_settings.max_cycles} (0 = unlimited)")
    click.echo(f"  Max worker iterations: {run_settings.max_worker_iterations}")
    click.echo(f"  Max task retries: {run_settings.max_task_retries}")
    click.echo(f"  Max retries (LLM): {run_settings.max_retries}")
    click.echo(f"  Request timeout: {run_settings.request_timeout}s")
    click.echo(f"  Command timeout: {run_settings.command_timeout}s")
    click.echo(f"  Test timeout: {run_settings.test_timeout}s")
    click.echo(f"  Max tokens: {run_settings.max_tokens}")
    click.echo(f"  Log level: {run_settings.log_level}")
    click.echo(f"  State file: {run_settings.state_file}")

    click.echo()
    click.echo("Test runners:")
    runner_found = False
    for name, binaries in _TEST_RUNNERS.items():
        available = any(_check_binary(b) for b in binaries)
        marker = "✓" if available else " "
        click.echo(f"  [{marker}] {name}")
        if available:
            runner_found = True
    if not runner_found:
        problems.append("No test runner binary found on PATH")

    click.echo()
    for line in _doctor_workspace(ws):
        click.echo(line)

    click.echo()
    if problems:
        click.secho("Issues found:", fg="yellow")
        for problem in problems:
            click.echo(f"  - {problem}")
        sys.exit(1)
    else:
        click.secho("All checks passed.", fg="green")


if __name__ == "__main__":
    main()
