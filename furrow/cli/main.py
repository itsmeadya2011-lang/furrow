"""Command line entry point for Furrow.

``start`` builds an explicit :class:`~furrow.config.Settings` instance from the
command line instead of mutating the module-level ``settings`` singleton, so the
web server and repeated in-process invocations never inherit stale overrides.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import click
from pydantic import ValidationError
from rich.console import Console

from furrow.config import Provider, Settings, settings
from furrow.core.orchestrator import Orchestrator

console = Console()

F = TypeVar("F", bound=Callable[..., Any])

_PROVIDERS = [provider.value for provider in Provider]


class ConfigError(Exception):
    """A problem the user can fix (bad option value, missing credential)."""


def _debug_option(func: F) -> F:
    return click.option(
        "-v",
        "--debug",
        is_flag=True,
        help="Show the full traceback when something goes wrong.",
    )(func)


def _echo_error(message: str) -> None:
    console.print(f"Error: {message}", style="bold red", markup=False, highlight=False)


def _hint_for(message: str) -> str | None:
    lowered = message.lower()
    if "api key" in lowered or "api_key" in lowered:
        return (
            "Set the provider key in the environment or a .env file "
            "(e.g. ANTHROPIC_API_KEY=... / OPENAI_API_KEY=...)."
        )
    if "workspace" in lowered:
        return "Pass --workspace pointing at an existing directory."
    return None


def _fail(exc: BaseException, debug: bool, context: str) -> None:
    """Print an actionable message (plus the traceback with ``--debug``) and exit 1."""
    message = str(exc) or exc.__class__.__name__
    _echo_error(message if isinstance(exc, ConfigError) else f"{context}: {message}")
    hint = _hint_for(message)
    if hint:
        console.print(hint, style="yellow", markup=False, highlight=False)
    if debug:
        console.print_exception()
    else:
        console.print("Re-run with -v/--debug for the full traceback.", style="dim")
    sys.exit(1)


def _resolve_goal(goal: str | None) -> str:
    """Accept the goal as an argument, prompt if missing, reject blanks."""
    text = goal
    if text is None:
        text = click.prompt("Enter your goal for Furrow")
    text = text.strip()
    if not text:
        raise ConfigError("Goal must not be empty.")
    return text


def _build_settings(
    provider: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: Path | None,
    test_command: str | None,
    state_file: Path | None,
) -> Settings:
    """A new ``Settings`` carrying the CLI overrides; the singleton is untouched."""
    data: dict[str, Any] = settings.model_dump()
    if provider is not None:
        data["provider"] = Provider(provider.strip().lower())
    if model is not None:
        if not model.strip():
            raise ConfigError("--model must not be blank.")
        data["model"] = model.strip()
    if max_cycles is not None:
        if max_cycles < 0:
            raise ConfigError("--max-cycles must be >= 0 (0 means unlimited).")
        data["max_cycles"] = max_cycles
    if max_parallel is not None:
        if max_parallel < 1:
            raise ConfigError("--max-parallel must be >= 1.")
        data["max_parallel_tasks"] = max_parallel
    if workspace is not None:
        if not workspace.is_dir():
            raise ConfigError(f"workspace does not exist or is not a directory: {workspace}")
        data["workspace"] = workspace
    if test_command is not None:
        if not test_command.strip():
            raise ConfigError("--test-command must not be blank.")
        data["test_command"] = test_command.strip()
    if state_file is not None:
        data["state_file"] = state_file
    try:
        return Settings(**data)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in error['loc']) or 'settings'}: {error['msg']}"
            for error in exc.errors()
        )
        raise ConfigError(f"invalid settings: {problems}") from exc


@click.group()
def main() -> None:
    """Furrow: an autonomous coding agent with an infinite parallel dev loop."""


@main.command()
@click.argument("goal", required=False)
@click.option(
    "--provider",
    type=click.Choice(_PROVIDERS, case_sensitive=False),
    default=None,
    help="LLM provider to use. Defaults to the configured provider (anthropic).",
)
@click.option("--model", default=None, help="Override the LLM model id.")
@click.option(
    "--max-cycles",
    type=int,
    default=None,
    help="Stop after this many cycles. 0 means unlimited (the default).",
)
@click.option(
    "--max-parallel",
    type=int,
    default=None,
    help="Maximum number of tasks executed in parallel.",
)
@click.option(
    "--workspace",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Directory the agents are allowed to edit. Defaults to the current directory.",
)
@click.option("--test-command", default=None, help="Command used to run the test suite.")
@click.option(
    "--state-file",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Where run state is persisted (JSON).",
)
@_debug_option
def start(
    goal: str | None,
    provider: str | None,
    model: str | None,
    max_cycles: int | None,
    max_parallel: int | None,
    workspace: Path | None,
    test_command: str | None,
    state_file: Path | None,
    debug: bool,
) -> None:
    """Run Furrow until the goal is done (or you stop it)."""
    try:
        resolved_goal = _resolve_goal(goal)
        run_settings = _build_settings(
            provider, model, max_cycles, max_parallel, workspace, test_command, state_file
        )
        console.print("Goal:", style="bold", end=" ")
        console.print(resolved_goal, markup=False, highlight=False)
        console.print(
            f"provider={run_settings.provider.value} model={run_settings.model} "
            f"max_cycles={run_settings.max_cycles} "
            f"max_parallel={run_settings.max_parallel_tasks} "
            f"workspace={run_settings.workspace}",
            style="dim",
            markup=False,
            highlight=False,
        )
        asyncio.run(Orchestrator(goal=resolved_goal, settings=run_settings).run())
    except KeyboardInterrupt:
        console.print("\nFurrow stopped by user.", style="yellow")
        sys.exit(0)
    except (click.Abort, click.ClickException):
        # ``click.Abort`` is a plain ``RuntimeError``, so without naming it here it
        # would be reported as a crash. Re-raising hands it back to click, which
        # prints ``Aborted!`` and exits non-zero -- the honest description of an
        # end-of-input on the goal prompt.
        raise
    except Exception as exc:  # noqa: BLE001 - every failure must surface as a message
        _fail(exc, debug, "run failed")
    sys.exit(0)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Interface to bind.")
@click.option(
    "--port",
    type=click.IntRange(1, 65535),
    default=8000,
    show_default=True,
    help="TCP port to bind.",
)
@_debug_option
def web(host: str, port: int, debug: bool) -> None:
    """Serve the web UI."""
    from furrow.web.server import run

    console.print(f"Serving Furrow on http://{host}:{port}", style="bold green")
    try:
        run(host=host, port=port)
    except KeyboardInterrupt:
        console.print("\nFurrow web server stopped by user.", style="yellow")
        sys.exit(0)
    except click.ClickException:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure must surface as a message
        _fail(exc, debug, f"could not serve on {host}:{port}")
    sys.exit(0)


if __name__ == "__main__":
    main()